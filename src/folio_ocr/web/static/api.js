// Calls to the folio server. No DOM. Errors are ApiError; a reply that arrives after its
// caller stopped caring is reported as STALE instead of data.
import { ApiError } from './errors.js';

export { ApiError };
export const STALE = Symbol('stale');

const enc = encodeURIComponent;

export const paths = {
  health: () => '/api/health',
  documents: () => '/api/documents',
  document: (id) => `/api/documents/${enc(id)}`,
  image: (id, n, v) => `/api/documents/${enc(id)}/pages/${enc(n)}/image${v > 0 ? `?v=${enc(v)}` : ''}`,
  read: (id, n) => `/api/documents/${enc(id)}/pages/${enc(n)}/read`,
  line: (id, lineId) => `/api/documents/${enc(id)}/lines/${enc(lineId)}`,
  exported: (id, fmt) => `/api/documents/${enc(id)}/export?format=${enc(fmt)}`,
};

async function send(fetchFn, url, init) {
  try {
    return await fetchFn(url, init);
  } catch {
    throw new ApiError('network');
  }
}

/** The server's one-line message from an error reply, or null when there is none. */
async function errorLine(res) {
  try {
    const body = await res.json();
    return typeof body?.error === 'string' ? body.error : null;
  } catch {
    return null;
  }
}

/** Read a body; a cut connection is `network`, unparsable JSON is `badBody`. */
async function readBody(res, read) {
  try {
    return await read(res);
  } catch (err) {
    if (err instanceof SyntaxError) throw new ApiError('badBody', res.status);
    throw new ApiError('network');
  }
}

/** Run one call; `read` turns a 2xx response into data (omit it for no body). */
async function call(fetchFn, url, init, isCurrent, read) {
  const res = await send(fetchFn, url, init);
  if (!isCurrent()) return STALE;
  if (!res.ok) {
    const line = await errorLine(res);
    if (!isCurrent()) return STALE;
    throw new ApiError('http', res.status, line);
  }
  const data = read ? await readBody(res, read) : undefined;
  return isCurrent() ? data : STALE;
}

const always = () => true;

export function callJson(fetchFn, method, url, { json, form, isCurrent = always } = {}) {
  const init = { method };
  if (json !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(json);
  } else if (form !== undefined) {
    init.body = form;
  }
  return call(fetchFn, url, init, isCurrent, (res) => res.json());
}

export function callBlob(fetchFn, url, { isCurrent = always } = {}) {
  return call(fetchFn, url, { method: 'GET' }, isCurrent, async (res) => ({ blob: await res.blob() }));
}

export async function callEmpty(fetchFn, method, url) {
  await call(fetchFn, url, { method }, always);
}
