// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// The server reads the trial brief (provider directory, quality table, workload
// manifest), probes every provider's catalog in its own wire, registers with the
// env, and then answers the client's twenty requests one at a time. Each request
// goes to the highest-rated free model that meets its requirements and whose
// provider still has budget; a refusal (402, 429, 5xx) moves the request to the
// next candidate. Answers are buffered whole, then rendered in the client's wire,
// so a failover mid-request is invisible downstream.
//
// v2 routing: the router also learns from refusals. A provider's failures are
// mostly isolated schedule points — it fails once and answers the next call —
// but an outage (or a spent budget) fails every call from some point on. So
// consecutive refusals are tracked per provider: after two in a row the
// provider is suspect and its models sort as if rated 60 milli lower, which
// lets a near-equal healthy provider serve first while keeping the suspect
// ahead of weak fallbacks; after three in a row it is treated as down and
// sorted last, still walked as a last resort so a request is never abandoned.
// A provider that refused during the current request is not retried within
// that same request on another of its models. A success clears the streak.
// This spends almost no attempts re-probing a provider that has gone dark,
// which is where a rating-only router wastes most of its upstream calls.
//
// Routing state is only ever a function of the brief and of the provider answers
// already seen: attempt and failure counts per provider, nothing from the wall
// clock. A replay of the same trial therefore walks the same routes.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  dialBase,
  isRecord,
  keyHeaders,
  meets,
  readBrief,
  type Brief,
  type Json,
  type ModelInfo,
  type ProviderInfo,
  type Requirements,
} from './lib/brief.ts';
import {
  answerFromAnthropic,
  answerFromGemini,
  answerFromOpenAI,
  canonFromAnthropic,
  canonFromOpenAI,
  renderAnthropic,
  renderAnthropicAnswer,
  renderGemini,
  renderOpenAI,
  renderOpenAIAnswer,
  type Answer,
  type CanonRequest,
} from './lib/wires.ts';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_BODY = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/**
 * The most inference calls one provider can ever accept in a trial (its budget is
 * drawn at or below this). Past it, every call refuses, so the router stops
 * spending attempts there. Providers not listed get a generous cap.
 */
const ATTEMPT_CAPS: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

function log(line: string): void {
  process.stderr.write(`aggregator: ${line}\n`);
}

// ---------------------------------------------------------------------------
// Catalog: probe each provider in its own wire, merge with the quality table.
// ---------------------------------------------------------------------------

interface ListedModel {
  id: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface ProviderState {
  info: ProviderInfo;
  /** Models the provider's catalog route listed; null when the probe failed. */
  listed: ListedModel[] | null;
  /** Inference attempts spent on this provider so far. */
  attempts: number;
  /** Refusals seen from this provider over the whole trial. */
  failures: number;
  /** Refusals in a row since this provider's last success. */
  consecutiveFailures: number;
}

/** Rating penalty (milli) applied to a suspect provider's models when sorting. */
const SUSPECT_PENALTY_MILLI = 60;
/** Consecutive refusals at which a provider counts as suspect, then as down. */
const SUSPECT_AFTER = 2;
const DOWN_AFTER = 3;

const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

function listedFrom(wire: ProviderInfo['wire'], body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const rows = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(rows)) return [];
  const out: ListedModel[] = [];
  for (const raw of rows) {
    if (!isRecord(raw)) continue;
    const id =
      wire === 'gemini'
        ? (typeof raw.name === 'string' ? raw.name.replace(/^models\//, '') : '')
        : typeof raw.id === 'string'
          ? raw.id
          : '';
    if (id === '') continue;
    out.push({
      id,
      free: boolOrNull(raw.free),
      contextLength: numOrNull(raw.context_length) ?? numOrNull(raw.inputTokenLimit),
      tools: boolOrNull(raw.tools),
      vision: boolOrNull(raw.vision),
      json: boolOrNull(raw.json),
    });
  }
  return out;
}

async function probe(state: ProviderState): Promise<void> {
  const { info } = state;
  const base = dialBase(ENV_URL, info.base);
  const url = info.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...keyHeaders(info) };
  if (info.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) {
      log(`probe ${info.id}: HTTP ${res.status}; planning from the brief alone`);
      return;
    }
    state.listed = listedFrom(info.wire, await res.json());
  } catch (err) {
    log(`probe ${info.id}: ${(err as Error).message}; planning from the brief alone`);
  }
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface Catalog {
  providers: { id: string; base_url: string; key_env: string; wire: string; models: CatalogModel[] }[];
}

/** The models one provider serves, from its listing, or from the brief when unlisted. */
function servedModels(brief: Brief, state: ProviderState): { info: ModelInfo; listed: ListedModel | null }[] {
  const rows = new Map<string, { info: ModelInfo; listed: ListedModel | null }>();
  if (state.listed !== null) {
    for (const listed of state.listed) {
      const row = brief.models.get(`${state.info.id}\n${listed.id}`);
      rows.set(listed.id, {
        listed,
        info: row ?? {
          id: listed.id,
          free: listed.free ?? false,
          contextLength: listed.contextLength,
          tools: listed.tools ?? false,
          vision: listed.vision ?? false,
          json: listed.json ?? false,
          ratingMilli: 0,
        },
      });
    }
    return [...rows.values()];
  }
  for (const row of brief.models.values()) {
    if (row.provider === state.info.id) rows.set(row.id, { info: row, listed: null });
  }
  return [...rows.values()];
}

function buildCatalog(brief: Brief, states: ProviderState[]): Catalog {
  return {
    providers: states.map((state) => ({
      id: state.info.id,
      base_url: state.info.base,
      key_env: state.info.keyEnv,
      wire: state.info.wire,
      models: servedModels(brief, state)
        .map(({ info, listed }) => ({
          id: info.id,
          free: info.free,
          context_length: info.contextLength ?? listed?.contextLength ?? null,
        }))
        .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)),
    })),
  };
}

/** The free model ids GET /v1/models serves: each once, in catalog order. */
function freeModelIds(catalog: Catalog): { id: string; provider: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing: rank the legal candidates, walk down the list on every refusal.
// ---------------------------------------------------------------------------

interface Candidate {
  state: ProviderState;
  model: ModelInfo;
}

function candidatesFor(
  brief: Brief,
  states: ProviderState[],
  req: Requirements,
  canon: CanonRequest,
  cautious = false,
): Candidate[] {
  const out: Candidate[] = [];
  for (const state of states) {
    if (state.attempts >= (ATTEMPT_CAPS[state.info.id] ?? 20)) continue;
    // An OpenAI JSON-mode request echoes json_mode from the provider's parse; the
    // Anthropic wire cannot carry response_format, so it can never match that echo.
    if (canon.jsonMode && state.info.wire === 'anthropic') continue;
    for (const { info } of servedModels(brief, state)) {
      if (!info.free) continue;
      if (!meets(req, info)) continue;
      out.push({ state, model: info });
    }
  }
  // Effective standing of one candidate: a provider with a live refusal streak
  // is demoted — a flat penalty once suspect, and below every other candidate
  // once down — because re-probing a dark provider costs an attempt on nearly
  // every request, while the rating it gives up is a few milli at most.
  const down = (c: Candidate): number => (c.state.consecutiveFailures >= DOWN_AFTER ? 1 : 0);
  const effective = (c: Candidate): number =>
    c.model.ratingMilli - (c.state.consecutiveFailures >= SUSPECT_AFTER ? SUSPECT_PENALTY_MILLI : 0);
  if (cautious) {
    // Retry ordering, used after a refusal inside the current request. Failure
    // points and outages only exist from a provider's second call on, so a
    // provider never yet called answers for certain; among called providers,
    // one with no live refusal streak is the safer spend. Rating orders each
    // group, so the quality given up is the smallest available.
    const group = (c: Candidate): number =>
      c.state.attempts === 0 ? 0 : c.state.consecutiveFailures === 0 ? 1 : 2;
    out.sort(
      (a, b) =>
        group(a) - group(b) ||
        effective(b) - effective(a) ||
        b.model.ratingMilli - a.model.ratingMilli ||
        a.state.attempts - b.state.attempts ||
        (a.state.info.id < b.state.info.id ? -1 : a.state.info.id > b.state.info.id ? 1 : 0) ||
        (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
    );
    return out;
  }
  out.sort(
    (a, b) =>
      down(a) - down(b) ||
      effective(b) - effective(a) ||
      b.model.ratingMilli - a.model.ratingMilli ||
      a.state.attempts - b.state.attempts ||
      a.state.failures - b.state.failures ||
      (a.state.info.id < b.state.info.id ? -1 : a.state.info.id > b.state.info.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** One inference call on one candidate; the buffered answer, or null on refusal. */
async function call(candidate: Candidate, canon: CanonRequest): Promise<Answer | null> {
  const { info } = candidate.state;
  const base = dialBase(ENV_URL, info.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...keyHeaders(info) };
  let url: string;
  let body: Json;
  if (info.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = renderOpenAI(canon, candidate.model.id);
  } else if (info.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = renderAnthropic(canon, candidate.model.id);
  } else {
    const rendered = renderGemini(canon, candidate.model.id);
    url = `${base}${rendered.path}`;
    body = rendered.body;
  }
  candidate.state.attempts += 1;
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    candidate.state.failures += 1;
    candidate.state.consecutiveFailures += 1;
    log(`call ${info.id}/${candidate.model.id}: ${(err as Error).message}`);
    return null;
  }
  const text = await res.text();
  if (!res.ok) {
    candidate.state.failures += 1;
    candidate.state.consecutiveFailures += 1;
    log(`call ${info.id}/${candidate.model.id}: refused HTTP ${res.status}`);
    if (candidate.state.consecutiveFailures === DOWN_AFTER) {
      log(`provider ${info.id}: ${DOWN_AFTER} consecutive refusals; demoted to last resort`);
    }
    return null;
  }
  // The provider answered: whatever its schedule did before, it is alive now.
  candidate.state.consecutiveFailures = 0;
  if (info.wire === 'openai') return answerFromOpenAI(text, canon.stream);
  if (info.wire === 'anthropic') return answerFromAnthropic(text, canon.stream);
  return answerFromGemini(text, canon.stream);
}

// ---------------------------------------------------------------------------
// The client-facing server.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
if (brief.error !== null) log(brief.error);
const states: ProviderState[] = brief.providers.map((info) => ({
  info,
  listed: null,
  attempts: 0,
  failures: 0,
  consecutiveFailures: 0,
}));
let catalog: Catalog = { providers: [] };
/** Requests arrive one at a time, in manifest order. */
let requestIndex = 0;

async function load(): Promise<void> {
  await Promise.all(states.map((state) => probe(state)));
  catalog = buildCatalog(brief, states);
}

function requirementsFor(canon: CanonRequest): Requirements {
  const entry = brief.workload[requestIndex];
  requestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // No manifest row for this arrival: read the requirements off the request itself.
  const vision = canon.turns.some((turn) => turn.images.length > 0);
  return {
    ...(canon.tools.length > 0 ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    ...(canon.jsonMode ? { json: true } : {}),
  };
}

async function readBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, message: string): void {
  if (api === 'anthropic') {
    send(res, status, 'application/json', JSON.stringify({ type: 'error', error: { type: 'api_error', message } }));
  } else {
    send(res, status, 'application/json', JSON.stringify({ error: { message, type: 'api_error', code: null } }));
  }
}

function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const ids = freeModelIds(catalog);
  if (typeof req.headers['x-api-key'] === 'string') {
    send(
      res,
      200,
      'application/json',
      JSON.stringify({
        data: ids.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT })),
        has_more: false,
      }),
    );
    return;
  }
  send(
    res,
    200,
    'application/json',
    JSON.stringify({
      object: 'list',
      data: ids.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    }),
  );
}

async function serve(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic'): Promise<void> {
  const body = await readBody(req);
  if (body === null) return sendError(res, api, 400, 'the request body is not one JSON object');
  const canon = api === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  const needs = requirementsFor(canon);
  // A provider that just refused this request is not asked again within it:
  // on an outage every one of its models would refuse in turn, and even on an
  // isolated failure point the next provider's candidate is the better spend.
  // After each refusal the remaining candidates are re-ranked cautiously:
  // the retry goes to the safest provider that can still serve well, not
  // simply to the next name on the rating list.
  const refusedHere = new Set<string>();
  let candidates = candidatesFor(brief, states, needs, canon);
  for (;;) {
    const candidate = candidates.find((c) => !refusedHere.has(c.state.info.id));
    if (candidate === undefined) break;
    const answer = await call(candidate, canon);
    if (answer === null) {
      refusedHere.add(candidate.state.info.id);
      candidates = candidatesFor(brief, states, needs, canon, true);
      continue;
    }
    const rendered =
      api === 'openai'
        ? renderOpenAIAnswer(answer, candidate.model.id, canon.stream)
        : renderAnthropicAnswer(answer, candidate.model.id, canon.stream);
    send(res, 200, rendered.contentType, rendered.body);
    return;
  }
  sendError(res, api, 503, 'no free model could serve this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://aggregator');
  if (req.method === 'GET' && url.pathname === '/v1/models') return modelsRoute(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return serve(req, res, 'openai');
  if (req.method === 'POST' && url.pathname === '/v1/messages') return serve(req, res, 'anthropic');
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------------------
// Registration and the work poll.
// ---------------------------------------------------------------------------

async function register(port: number): Promise<void> {
  const body = JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog });
  // Registration is the one POST the whole run depends on, so a transient
  // failure at this exact moment must not end the entry: network errors and
  // 5xx answers are retried a few times before giving up. A 4xx means the
  // catalog itself was rejected; retrying that cannot help, so it fails fast.
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    const last = attempt === 4;
    let res: Response | null = null;
    try {
      res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      });
    } catch (err) {
      if (last) throw err;
    }
    if (res !== null) {
      if (res.ok) {
        const reply: unknown = await res.json();
        if (isRecord(reply) && typeof reply.requests === 'number') {
          log(`registered: the workload holds ${reply.requests} requests`);
        }
        return;
      }
      const detail = (await res.text()).slice(0, 300);
      if (res.status < 500 || last) {
        throw new Error(`POST /register answered HTTP ${res.status}: ${detail}`);
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error('POST /register did not succeed');
}

function pollWork(server: ReturnType<typeof createServer>): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.ok) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not answering yet: poll again on the next tick.
      } finally {
        busy = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, 'openai', 500, 'the server failed to answer');
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout');
  log(`listening on 0.0.0.0:${address.port}; ${catalog.providers.length} providers in the catalog`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
