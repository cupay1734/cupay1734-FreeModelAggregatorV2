// The trial brief: three JSON documents inside one text file, found by their schema tag.
//
//   providers/v1  the directory: ten free-tier providers (id, wire, auth, key, base path)
//   ratings/v1    the quality table: per provider+model rating, capabilities, free flag
//   workload/v1   the manifest: the requests the client will send, in order, with the
//                 requirements a model must meet to serve each one
//
// The brief is the only input the server plans from. Everything derived from it is a pure
// function of its bytes, so the catalog printed by --list and the one sent to /register
// are the same document in every run and every replay.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  /** The provider's own path as the directory spells it, e.g. /p/groq. */
  base: string;
  keyEnv: string;
  key: string;
}

export interface ModelInfo {
  id: string;
  free: boolean;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  ratingMilli: number;
}

export interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  requires: Requirements;
}

export interface Brief {
  providers: ProviderInfo[];
  /** Ratings rows keyed `${providerId}\n${modelId}`: the sealed capability truth. */
  models: Map<string, ModelInfo & { provider: string }>;
  workload: WorkloadEntry[];
  error: string | null;
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

/** Every top-level JSON object in `text` carrying a string `schema`, in file order. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a standalone document: keep scanning.
    }
  }
  return found;
}

/** Index of the brace closing the object opened at `start`, skipping string contents. */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function providerFrom(raw: unknown): ProviderInfo | null {
  if (!isRecord(raw)) return null;
  const id = str(raw.id);
  const wire = raw.wire === 'openai' || raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    raw.auth === 'bearer' || raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none'
      ? raw.auth
      : 'bearer';
  const base =
    str(raw.path) || str(raw.base_path) || str(raw.base_url) || str(raw.base);
  return {
    id,
    name: str(raw.name) || id,
    wire,
    auth,
    base,
    keyEnv: str(raw.key_env),
    key: str(raw.key),
  };
}

export function readBrief(path: string): Brief {
  const brief: Brief = { providers: [], models: new Map(), workload: [], error: null };
  if (path === '') {
    brief.error = 'MAKEFUN_PROMPT_FILE is not set';
    return brief;
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    brief.error = `cannot read the brief at ${path}: ${(err as Error).message}`;
    return brief;
  }
  const docs = schemaDocuments(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) {
    brief.error = `the brief holds no ${PROVIDERS_SCHEMA} document`;
    return brief;
  }
  brief.providers = (directory.providers as unknown[]).flatMap((raw) => {
    const p = providerFrom(raw);
    return p === null ? [] : [p];
  });

  const ratings = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  if (ratings !== undefined) {
    for (const raw of ratings.ratings as unknown[]) {
      if (!isRecord(raw)) continue;
      const provider = str(raw.provider);
      const model = str(raw.model);
      if (provider === '' || model === '') continue;
      brief.models.set(`${provider}\n${model}`, {
        provider,
        id: model,
        free: raw.free === true,
        contextLength: numOrNull(raw.context_length),
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        ratingMilli: numOrNull(raw.rating_milli) ?? 0,
      });
    }
  }

  const workload = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  if (workload !== undefined) {
    brief.workload = (workload.manifest as unknown[]).flatMap((raw) => {
      if (!isRecord(raw) || str(raw.id) === '') return [];
      const req = isRecord(raw.requires) ? raw.requires : {};
      const entry: WorkloadEntry = {
        id: str(raw.id),
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        requires: {
          ...(req.tools === true ? { tools: true } : {}),
          ...(req.vision === true ? { vision: true } : {}),
          ...(req.json === true ? { json: true } : {}),
          ...(numOrNull(req.min_context) !== null ? { minContext: numOrNull(req.min_context) as number } : {}),
        },
      };
      return [entry];
    });
  }
  return brief;
}

/** Whether a model with these capabilities may serve a request with these requirements. */
export function meets(req: Requirements, m: ModelInfo): boolean {
  if (req.tools === true && !m.tools) return false;
  if (req.vision === true && !m.vision) return false;
  if (req.json === true && !m.json) return false;
  if (req.minContext !== undefined && (m.contextLength === null || m.contextLength < req.minContext)) return false;
  return true;
}

/** The absolute URL a provider hangs off: its directory path joined to the env's URL. */
export function dialBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const root = envUrl.replace(/\/+$/, '');
  if (b === '') return root;
  return `${root}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers carrying a provider's key, in the scheme the directory names. */
export function keyHeaders(provider: ProviderInfo): Record<string, string> {
  if (provider.key === '' || provider.auth === 'none') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}
