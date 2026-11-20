import test from 'node:test';
import assert from 'node:assert/strict';
import { createReadQueue } from '../../src/folio_ocr/web/static/queue.js';

const tick = () => new Promise((s) => setImmediate(s));

/** Fake server: each fetch is parked until the test answers it. */
function harness(overrides = {}) {
  const calls = [];
  const log = { done: [], errors: [], starts: [], changes: [] };
  let seq = 0;
  const fetchFn = (url) => new Promise((resolve) => { calls.push({ url, resolve }); });
  const alive = new Set(['a', 'b']);
  const q = createReadQueue({
    fetchFn,
    isAlive: (id) => alive.has(id),
    stamp: () => ++seq,
    onChange: (s) => log.changes.push(s),
    onStart: (i) => log.starts.push(i),
    onDone: (i, page, readSeq) => log.done.push({ i, page, readSeq }),
    onError: (i, err) => log.errors.push({ i, err }),
    ...overrides,
  });
  return { q, calls, log, alive };
}

/** Answer a parked fetch: headers now, body settled by the returned function. */
function answer(call, status, body) {
  let settle;
  const p = new Promise((res) => { settle = () => res(body); });
  call.resolve({ ok: status < 300, status, headers: new Headers(), json: () => p });
  return settle;
}

const item = (docId, page) => ({ docId, page });
const pages = (log) => log.done.map((d) => `${d.i.docId}${d.i.page}`);

test('only one read is sent until the first body has arrived', async () => {
  const { q, calls } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  assert.equal(calls.length, 1);
  const settle = answer(calls[0], 200, { index: 1 });
  await tick();
  assert.equal(calls.length, 1, 'body still pending');
  settle();
  await tick();
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, '/api/documents/a/pages/2/read');
});

test('reads run first in, first out and report onStart', async () => {
  const { q, calls, log } = harness();
  q.enqueue([item('a', 1), item('b', 1), item('a', 2)]);
  for (let i = 0; i < 3; i += 1) {
    await tick();
    answer(calls[i], 200, { n: i })();
  }
  await tick();
  assert.deepEqual(pages(log), ['a1', 'b1', 'a2']);
  assert.deepEqual(log.starts, [item('a', 1), item('b', 1), item('a', 2)]);
  assert.deepEqual(log.done[0].page, { n: 0 });
});

test('front() makes a queued page next', async () => {
  const { q, calls } = harness();
  q.enqueue([item('a', 1), item('a', 2), item('a', 3)]);
  await tick();
  q.front(item('a', 3));
  answer(calls[0], 200, {})();
  await tick();
  assert.equal(calls[1].url, '/api/documents/a/pages/3/read');
  assert.deepEqual(q.snapshot().queue, [item('a', 2)]);
});

test('front() inserts an item that is not queued', async () => {
  const { q } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  q.front(item('b', 9));
  assert.deepEqual(q.snapshot().queue, [item('b', 9), item('a', 2)]);
});

test('enqueue skips queued and in-flight items', async () => {
  const { q } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  q.enqueue([item('a', 1), item('a', 2), item('a', 2), item('a', 3)]);
  assert.deepEqual(q.snapshot().queue, [item('a', 2), item('a', 3)]);
  assert.equal(q.snapshot().inFlight.page, 1);
});

test('409 pauses with the item at the front; resume() sends it again', async () => {
  const { q, calls, log } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  answer(calls[0], 409, { error: 'busy' })();
  await tick();
  assert.equal(log.errors[0].err.status, 409);
  assert.equal(calls.length, 1);
  const s = q.snapshot();
  assert.equal(s.paused, true);
  assert.equal(s.inFlight, null);
  assert.deepEqual(s.queue, [item('a', 1), item('a', 2)]);
  q.resume();
  await tick();
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, '/api/documents/a/pages/1/read');
  assert.equal(q.snapshot().paused, false);
});

test('after a 409, stop() clears the pause so a new enqueue is sent', async () => {
  const { q, calls } = harness();
  q.enqueue([item('a', 1)]);
  await tick();
  answer(calls[0], 409, { error: 'busy' })();
  await tick();
  q.stop();
  assert.equal(q.snapshot().paused, false);
  q.enqueue([item('b', 1)]);
  await tick();
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, '/api/documents/b/pages/1/read');
});

test('after a 409, enqueue alone sends nothing until front() or resume()', async () => {
  const { q, calls } = harness();
  q.enqueue([item('a', 1)]);
  await tick();
  answer(calls[0], 409, { error: 'busy' })();
  await tick();
  q.enqueue([item('b', 1)]);
  await tick();
  assert.equal(calls.length, 1);
  assert.equal(q.snapshot().paused, true);
  assert.deepEqual(q.snapshot().queue, [item('a', 1), item('b', 1)]);
});

test('front() unpauses', async () => {
  const { q, calls } = harness();
  q.enqueue([item('a', 1)]);
  await tick();
  answer(calls[0], 409, { error: 'busy' })();
  await tick();
  q.front(item('a', 1));
  await tick();
  assert.equal(calls.length, 2);
});

test('400 skips the page and continues', async () => {
  const { q, calls, log } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  answer(calls[0], 400, { error: 'bad' })();
  await tick();
  assert.equal(log.errors[0].err.serverLine, 'bad');
  assert.equal(calls.length, 2);
});

test('404 drops that document but keeps others', async () => {
  const { q, calls, log } = harness();
  q.enqueue([item('a', 1), item('a', 2), item('b', 1)]);
  await tick();
  answer(calls[0], 404, { error: 'gone' })();
  await tick();
  assert.equal(log.errors.length, 1);
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, '/api/documents/b/pages/1/read');
});

test('a network error empties the queue', async () => {
  const { q, calls, log } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  calls[0].resolve(Promise.reject(new TypeError('down')));
  await tick();
  assert.equal(log.errors[0].err.kind, 'network');
  assert.deepEqual(q.snapshot().queue, []);
  assert.equal(calls.length, 1);
});

test('a 500 empties the queue', async () => {
  const { q, calls, log } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  answer(calls[0], 500, { error: 'x' })();
  await tick();
  assert.equal(log.errors.length, 1);
  assert.deepEqual(q.snapshot().queue, []);
});

test('stop() empties the queue and the in-flight result is still delivered', async () => {
  const { q, calls, log } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  q.stop();
  assert.deepEqual(q.snapshot().queue, []);
  answer(calls[0], 200, { n: 1 })();
  await tick();
  assert.deepEqual(pages(log), ['a1']);
  assert.equal(calls.length, 1);
});

test('a document deleted while the body is pending is dropped and the queue goes on', async () => {
  const { q, calls, log, alive } = harness();
  q.enqueue([item('a', 1), item('b', 1)]);
  await tick();
  const settle = answer(calls[0], 200, { n: 1 });
  await tick();
  alive.delete('a');
  settle();
  await tick();
  assert.equal(log.done.length, 0);
  assert.equal(log.errors.length, 0);
  assert.equal(calls.length, 2);
});

test('readSeq is the stamp taken right before that fetch', async () => {
  const { q, calls, log } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  answer(calls[0], 200, {})();
  await tick();
  answer(calls[1], 200, {})();
  await tick();
  assert.deepEqual(log.done.map((d) => d.readSeq), [1, 2]);
});

test('dropDoc removes queued items of that document', async () => {
  const { q } = harness();
  q.enqueue([item('a', 1), item('b', 1), item('a', 2)]);
  await tick();
  q.dropDoc('a');
  assert.deepEqual(q.snapshot().queue, [item('b', 1)]);
});

test('snapshot returns copies and onChange fires with them', async () => {
  const { q, log } = harness();
  q.enqueue([item('a', 1), item('a', 2)]);
  await tick();
  const s = q.snapshot();
  s.queue.push(item('x', 1));
  s.inFlight.page = 99;
  const again = q.snapshot();
  assert.deepEqual(again.queue, [item('a', 2)]);
  assert.equal(again.inFlight.page, 1);
  assert.equal(typeof again.inFlight.startedAt, 'number');
  assert.ok(log.changes.length >= 2);
});
