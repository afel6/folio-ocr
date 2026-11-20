import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ApiError, STALE, paths, callJson, callBlob, callEmpty,
} from '../../src/folio_ocr/web/static/api.js';

function deferred() {
  let resolve; let reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}

/** A response whose body promise is settled separately from the headers. */
function reply(status, body, extra = {}) {
  const d = deferred();
  const res = {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers(extra.headers || {}),
    json: () => d.promise,
    blob: () => d.promise,
  };
  return { res, body: d, settle: () => d.resolve(body) };
}

const fakeFetch = (res) => async () => res;

async function caught(promise) {
  try { await promise; } catch (e) { return e; }
  return null;
}

test('callJson maps a rejected fetch to network', async () => {
  const err = await caught(callJson(async () => { throw new TypeError('x'); }, 'GET', '/u'));
  assert.ok(err instanceof ApiError);
  assert.equal(err.kind, 'network');
});

test('callJson 404 carries the server line', async () => {
  const r = reply(404, { error: 'x' }); r.settle();
  const err = await caught(callJson(fakeFetch(r.res), 'GET', '/u'));
  assert.equal(err.kind, 'http');
  assert.equal(err.status, 404);
  assert.equal(err.serverLine, 'x');
});

test('callJson error with unparsable body has null serverLine', async () => {
  const r = reply(500, null); r.body.reject(new SyntaxError('bad'));
  const err = await caught(callJson(fakeFetch(r.res), 'GET', '/u'));
  assert.equal(err.status, 500);
  assert.equal(err.serverLine, null);
});

test('callJson error with non-string error field has null serverLine', async () => {
  const r = reply(400, { error: 5 }); r.settle();
  const err = await caught(callJson(fakeFetch(r.res), 'GET', '/u'));
  assert.equal(err.serverLine, null);
});

test('callJson 2xx SyntaxError is badBody with status', async () => {
  const r = reply(200, null); r.body.reject(new SyntaxError('bad'));
  const err = await caught(callJson(fakeFetch(r.res), 'GET', '/u'));
  assert.equal(err.kind, 'badBody');
  assert.equal(err.status, 200);
});

test('callJson 2xx TypeError while reading is network', async () => {
  const r = reply(200, null); r.body.reject(new TypeError('cut'));
  const err = await caught(callJson(fakeFetch(r.res), 'GET', '/u'));
  assert.equal(err.kind, 'network');
});

test('callJson returns the parsed data untouched', async () => {
  const r = reply(200, [1, 'a']); r.settle();
  assert.deepEqual(await callJson(fakeFetch(r.res), 'GET', '/u'), [1, 'a']);
});

test('callJson is STALE when not current after headers', async () => {
  const r = reply(200, {}); r.settle();
  const out = await callJson(fakeFetch(r.res), 'GET', '/u', { isCurrent: () => false });
  assert.equal(out, STALE);
});

test('callJson is STALE when isCurrent flips while the body is pending', async () => {
  const r = reply(200, { ok: 1 });
  let current = true;
  const p = callJson(fakeFetch(r.res), 'GET', '/u', { isCurrent: () => current });
  await new Promise((s) => setImmediate(s));
  current = false;
  r.settle();
  assert.equal(await p, STALE);
});

test('callJson stale error reply with current false is STALE too', async () => {
  const r = reply(404, { error: 'gone' }); r.settle();
  const out = await callJson(fakeFetch(r.res), 'GET', '/u', { isCurrent: () => false });
  assert.equal(out, STALE);
});

test('callJson sends exact JSON request', async () => {
  let seen;
  const r = reply(200, {}); r.settle();
  await callJson(async (url, init) => { seen = { url, init }; return r.res; }, 'PUT', '/u', {
    json: { final_text: 'é' },
  });
  assert.equal(seen.init.method, 'PUT');
  assert.deepEqual(seen.init.headers, { 'Content-Type': 'application/json' });
  assert.equal(seen.init.body, JSON.stringify({ final_text: 'é' }));
});

test('callJson form request sets no content type', async () => {
  let seen;
  const form = new FormData();
  const r = reply(200, {}); r.settle();
  await callJson(async (url, init) => { seen = init; return r.res; }, 'POST', '/u', { form });
  assert.equal(seen.body, form);
  assert.equal(seen.headers, undefined);
});

test('paths encode ids and add ?v only above zero', () => {
  assert.equal(paths.health(), '/api/health');
  assert.equal(paths.documents(), '/api/documents');
  assert.equal(paths.document('a/b'), '/api/documents/a%2Fb');
  assert.equal(paths.image('a b', 2, 0), '/api/documents/a%20b/pages/2/image');
  assert.equal(paths.image('d', 2, 3), '/api/documents/d/pages/2/image?v=3');
  assert.equal(paths.read('d', 4), '/api/documents/d/pages/4/read');
  assert.equal(paths.line('d', 'p1-l2'), '/api/documents/d/lines/p1-l2');
  assert.equal(paths.exported('d', 'txt'), '/api/documents/d/export?format=txt');
});

test('callBlob returns just the blob', async () => {
  const r = reply(200, 'BLOB'); r.settle();
  const out = await callBlob(fakeFetch(r.res), '/u');
  assert.deepEqual(out, { blob: 'BLOB' });
});

test('callBlob maps a rejected blob() to network', async () => {
  const r = reply(200, null); r.body.reject(new TypeError('cut'));
  const err = await caught(callBlob(fakeFetch(r.res), '/u'));
  assert.equal(err.kind, 'network');
});

test('callBlob is STALE after the body when not current', async () => {
  const r = reply(200, 'B');
  let current = true;
  const p = callBlob(fakeFetch(r.res), '/u', { isCurrent: () => current });
  await new Promise((s) => setImmediate(s));
  current = false;
  r.settle();
  assert.equal(await p, STALE);
});

test('callBlob error status throws http', async () => {
  const r = reply(400, { error: 'bad' }); r.settle();
  const err = await caught(callBlob(fakeFetch(r.res), '/u'));
  assert.equal(err.status, 400);
  assert.equal(err.serverLine, 'bad');
});

test('callEmpty accepts 204', async () => {
  const r = reply(204, null);
  assert.equal(await callEmpty(fakeFetch(r.res), 'DELETE', '/u'), undefined);
});

test('callEmpty 404 throws http', async () => {
  const r = reply(404, { error: 'nope' }); r.settle();
  const err = await caught(callEmpty(fakeFetch(r.res), 'DELETE', '/u'));
  assert.equal(err.status, 404);
  assert.equal(err.serverLine, 'nope');
});
