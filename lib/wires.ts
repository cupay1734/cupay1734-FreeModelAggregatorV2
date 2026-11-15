// Wire translation: client wires (OpenAI, Anthropic) <-> provider wires (OpenAI,
// Anthropic, Gemini), through one canonical request form.
//
// The providers answer with an echo of what they parsed: system text, the last user
// text, tool names, max_tokens, stop sequences, image bytes and the JSON-mode flag.
// The canonical form carries exactly those fields untouched, and each renderer puts
// them where that wire's parser reads them, so an echo survives any translation.
// Answers come back as one buffered {text, usage} and are rendered into the client's
// wire, streamed or not: buffering first is what makes failover invisible to the
// client, and the client's checks read frame order and totals, never timing.

import { isRecord, type Json } from './brief.ts';

export interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

interface ImageRef {
  mediaType: string;
  /** Base64 payload when the image travels inline; otherwise the source URL. */
  data: string | null;
  url: string | null;
}

interface Turn {
  role: 'user' | 'assistant';
  text: string;
  images: ImageRef[];
  toolCalls: { id: string; name: string; argsJson: string }[];
  toolResults: { id: string; text: string }[];
}

export interface CanonRequest {
  system: string;
  turns: Turn[];
  tools: { name: string; description: string; schema: Json }[];
  toolChoice: { kind: 'auto' | 'any' | 'tool' | 'none'; name?: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

const strList = (v: unknown): string[] => {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
};

/** The text of a string or of an array's text blocks/parts, joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function imageFromDataUrl(url: string): ImageRef | null {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (match === null) return { mediaType: 'image/png', data: null, url };
  if (match[2] !== undefined) return { mediaType: match[1] || 'image/png', data: match[3], url: null };
  return { mediaType: 'image/png', data: null, url };
}

function dataUrlOf(image: ImageRef): string {
  if (image.url !== null) return image.url;
  return `data:${image.mediaType};base64,${image.data ?? ''}`;
}

// ---------------------------------------------------------------------------
// Client request -> canonical form
// ---------------------------------------------------------------------------

export function canonFromOpenAI(body: Json): CanonRequest {
  const canon: CanonRequest = {
    system: '',
    turns: [],
    tools: [],
    toolChoice: null,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: strList(body.stop),
    jsonMode:
      isRecord(body.response_format) &&
      (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema'),
    stream: body.stream === true,
  };
  const systemParts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      canon.turns.push({
        role: 'user',
        text: '',
        images: [],
        toolCalls: [],
        toolResults: [
          {
            id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            text: typeof raw.content === 'string' ? raw.content : blocksText(raw.content),
          },
        ],
      });
      continue;
    }
    const turn: Turn = {
      role: raw.role === 'assistant' ? 'assistant' : 'user',
      text: blocksText(raw.content),
      images: [],
      toolCalls: [],
      toolResults: [],
    };
    if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const image = imageFromDataUrl(part.image_url.url);
          if (image !== null) turn.images.push(image);
        }
      }
    }
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        turn.toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          argsJson: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
        });
      }
    }
    canon.turns.push(turn);
  }
  canon.system = systemParts.join('\n');
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      canon.tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  if (choice === 'auto') canon.toolChoice = { kind: 'auto' };
  else if (choice === 'required') canon.toolChoice = { kind: 'any' };
  else if (choice === 'none') canon.toolChoice = { kind: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    canon.toolChoice = { kind: 'tool', name: choice.function.name };
  }
  return canon;
}

export function canonFromAnthropic(body: Json): CanonRequest {
  const canon: CanonRequest = {
    system: blocksText(body.system),
    turns: [],
    tools: [],
    toolChoice: null,
    maxTokens: numOrNull(body.max_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: strList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const turn: Turn = {
      role: raw.role === 'assistant' ? 'assistant' : 'user',
      text: blocksText(raw.content),
      images: [],
      toolCalls: [],
      toolResults: [],
    };
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            turn.images.push({
              mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
              data: block.source.data,
              url: null,
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            turn.images.push({ mediaType: 'image/png', data: null, url: block.source.url });
          }
        } else if (block.type === 'tool_use') {
          turn.toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            argsJson: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
        } else if (block.type === 'tool_result') {
          turn.toolResults.push({
            id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            text:
              typeof block.content === 'string'
                ? block.content
                : blocksText(block.content) || JSON.stringify(block.content ?? ''),
          });
        }
      }
    }
    canon.turns.push(turn);
  }
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      canon.tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') canon.toolChoice = { kind: 'auto' };
    else if (choice.type === 'any') canon.toolChoice = { kind: 'any' };
    else if (choice.type === 'none') canon.toolChoice = { kind: 'none' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') canon.toolChoice = { kind: 'tool', name: choice.name };
  }
  return canon;
}

// ---------------------------------------------------------------------------
// Canonical form -> provider wire
// ---------------------------------------------------------------------------

export function renderOpenAI(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
  for (const turn of canon.turns) {
    for (const result of turn.toolResults) {
      messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
    }
    if (turn.role === 'assistant') {
      messages.push({
        role: 'assistant',
        content: turn.text !== '' ? turn.text : null,
        ...(turn.toolCalls.length > 0
          ? {
              tool_calls: turn.toolCalls.map((call) => ({
                id: call.id,
                type: 'function',
                function: { name: call.name, arguments: call.argsJson },
              })),
            }
          : {}),
      });
      continue;
    }
    if (turn.images.length === 0) {
      if (turn.text !== '' || turn.toolResults.length === 0) messages.push({ role: 'user', content: turn.text });
      continue;
    }
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ type: 'text', text: turn.text });
    for (const image of turn.images) parts.push({ type: 'image_url', image_url: { url: dataUrlOf(image) } });
    messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop = canon.stops;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.schema },
    }));
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.kind === 'auto') out.tool_choice = 'auto';
    else if (canon.toolChoice.kind === 'any') out.tool_choice = 'required';
    else if (canon.toolChoice.kind === 'none') out.tool_choice = 'none';
    else if (canon.toolChoice.kind === 'tool') {
      out.tool_choice = { type: 'function', function: { name: canon.toolChoice.name ?? '' } };
    }
  }
  if (canon.jsonMode) out.response_format = { type: 'json_object' };
  if (canon.stream) {
    out.stream = true;
    // Ask for the closing usage chunk: the client scores the usage the provider reports,
    // and a stream without one carries none.
    out.stream_options = { include_usage: true };
  }
  return out;
}

export function renderAnthropic(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of canon.turns) {
    if (turn.role === 'assistant') {
      const blocks: Json[] = [];
      if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
      for (const call of turn.toolCalls) {
        let input: unknown = {};
        try {
          input = JSON.parse(call.argsJson);
        } catch {
          input = {};
        }
        blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: isRecord(input) ? input : {} });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    const blocks: Json[] = [];
    if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
    for (const image of turn.images) {
      blocks.push(
        image.url !== null
          ? { type: 'image', source: { type: 'url', url: image.url } }
          : { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data ?? '' } },
      );
    }
    for (const result of turn.toolResults) {
      blocks.push({ type: 'tool_result', tool_use_id: result.id, content: result.text });
    }
    messages.push({ role: 'user', content: blocks.length > 0 ? blocks : '' });
  }
  const out: Json = { model, messages };
  if (canon.system !== '') out.system = canon.system;
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop_sequences = canon.stops;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.schema }));
  }
  if (canon.toolChoice !== null) {
    if (canon.toolChoice.kind === 'auto') out.tool_choice = { type: 'auto' };
    else if (canon.toolChoice.kind === 'any') out.tool_choice = { type: 'any' };
    else if (canon.toolChoice.kind === 'none') out.tool_choice = { type: 'none' };
    else if (canon.toolChoice.kind === 'tool') out.tool_choice = { type: 'tool', name: canon.toolChoice.name ?? '' };
  }
  if (canon.stream) out.stream = true;
  return out;
}

export function renderGemini(canon: CanonRequest, model: string): { path: string; body: Json } {
  const contents: Json[] = [];
  for (const turn of canon.turns) {
    const parts: Json[] = [];
    if (turn.role === 'assistant') {
      if (turn.text !== '') parts.push({ text: turn.text });
      for (const call of turn.toolCalls) {
        let args: unknown = {};
        try {
          args = JSON.parse(call.argsJson);
        } catch {
          args = {};
        }
        parts.push({ functionCall: { name: call.name, args: isRecord(args) ? args : {} } });
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (turn.text !== '') parts.push({ text: turn.text });
    for (const image of turn.images) {
      parts.push(
        image.url !== null
          ? { fileData: { fileUri: image.url } }
          : { inlineData: { mimeType: image.mediaType, data: image.data ?? '' } },
      );
    }
    for (const result of turn.toolResults) {
      parts.push({ functionResponse: { name: '', response: { result: result.text } } });
    }
    contents.push({ role: 'user', parts });
  }
  const body: Json = { contents };
  if (canon.system !== '') body.systemInstruction = { parts: [{ text: canon.system }] };
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.topP !== null) config.topP = canon.topP;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  if (canon.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: canon.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.schema,
        })),
      },
    ];
  }
  const verb = canon.stream ? 'streamGenerateContent' : 'generateContent';
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${verb}`, body };
}

// ---------------------------------------------------------------------------
// Provider wire -> buffered answer
// ---------------------------------------------------------------------------

/** The data payloads of an SSE body, in order. */
function ssePayloads(text: string): string[] {
  const payloads: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) payloads.push(line.slice(5).replace(/^ /, ''));
    }
  }
  return payloads;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

const zeroUsage = { inputTokens: 0, outputTokens: 0 };

function usagePair(usage: Json | undefined, inputKey: string, outputKey: string): { inputTokens: number; outputTokens: number } {
  if (usage === undefined) return zeroUsage;
  return {
    inputTokens: numOrNull(usage[inputKey]) ?? 0,
    outputTokens: numOrNull(usage[outputKey]) ?? 0,
  };
}

export function answerFromOpenAI(text: string, stream: boolean): Answer {
  if (!stream) {
    const body = jsonOrNull(text) ?? {};
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const content =
      typeof message.content === 'string' ? message.content : blocksText(message.content);
    return { text: content, ...usagePair(isRecord(body.usage) ? body.usage : undefined, 'prompt_tokens', 'completion_tokens') };
  }
  let out = '';
  let usage = zeroUsage;
  for (const payload of ssePayloads(text)) {
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) usage = usagePair(chunk.usage, 'prompt_tokens', 'completion_tokens');
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null && isRecord(choice.delta) && typeof choice.delta.content === 'string') {
      out += choice.delta.content;
    }
  }
  return { text: out, ...usage };
}

export function answerFromAnthropic(text: string, stream: boolean): Answer {
  if (!stream) {
    const body = jsonOrNull(text) ?? {};
    const parts: string[] = [];
    if (Array.isArray(body.content)) {
      for (const block of body.content) {
        if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
      }
    }
    return { text: parts.join(''), ...usagePair(isRecord(body.usage) ? body.usage : undefined, 'input_tokens', 'output_tokens') };
  }
  let out = '';
  let usage = zeroUsage;
  for (const payload of ssePayloads(text)) {
    const event = jsonOrNull(payload);
    if (event === null) continue;
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      usage = { ...usage, inputTokens: numOrNull(event.message.usage.input_tokens) ?? 0 };
    } else if (event.type === 'content_block_delta' && isRecord(event.delta) && event.delta.type === 'text_delta') {
      if (typeof event.delta.text === 'string') out += event.delta.text;
    } else if (event.type === 'message_delta' && isRecord(event.usage)) {
      usage = { ...usage, outputTokens: numOrNull(event.usage.output_tokens) ?? 0 };
    }
  }
  return { text: out, ...usage };
}

export function answerFromGemini(text: string, stream: boolean): Answer {
  const fromFrame = (frame: Json, into: Answer): void => {
    const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null && isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
      for (const part of candidate.content.parts) {
        if (isRecord(part) && typeof part.text === 'string') into.text += part.text;
      }
    }
    if (isRecord(frame.usageMetadata)) {
      into.inputTokens = numOrNull(frame.usageMetadata.promptTokenCount) ?? into.inputTokens;
      into.outputTokens = numOrNull(frame.usageMetadata.candidatesTokenCount) ?? into.outputTokens;
    }
  };
  const answer: Answer = { text: '', inputTokens: 0, outputTokens: 0 };
  if (!stream) {
    const body = jsonOrNull(text);
    if (body !== null) fromFrame(body, answer);
    return answer;
  }
  for (const payload of ssePayloads(text)) {
    const frame = jsonOrNull(payload);
    if (frame !== null) fromFrame(frame, answer);
  }
  return answer;
}

// ---------------------------------------------------------------------------
// Buffered answer -> client wire
// ---------------------------------------------------------------------------

const FIXED_CREATED = 1767225600;
const TEXT_PIECE = 13;

/** The answer text in fixed-size pieces, as the simulated providers stream it. */
function pieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += TEXT_PIECE) out.push(text.slice(at, at + TEXT_PIECE));
  return out;
}

function sse(data: unknown, event: string | null = null): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export function renderOpenAIAnswer(answer: Answer, model: string, stream: boolean): { contentType: string; body: string } {
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregator',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop' }],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sse(chunk({ role: 'assistant', content: '' }, null));
  for (const piece of pieces(answer.text)) out += sse(chunk({ content: piece }, null));
  out += sse(chunk({}, 'stop'));
  out += sse({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage,
  });
  out += 'data: [DONE]\n\n';
  return { contentType: 'text/event-stream', body: out };
}

export function renderAnthropicAnswer(answer: Answer, model: string, stream: boolean): { contentType: string; body: string } {
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
      }),
    };
  }
  let out = sse(
    {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    },
    'message_start',
  );
  out += sse({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }, 'content_block_start');
  for (const piece of pieces(answer.text)) {
    out += sse({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }, 'content_block_delta');
  }
  out += sse({ type: 'content_block_stop', index: 0 }, 'content_block_stop');
  out += sse(
    {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    },
    'message_delta',
  );
  out += sse({ type: 'message_stop' }, 'message_stop');
  return { contentType: 'text/event-stream', body: out };
}
