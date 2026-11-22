import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { blankState, installPage, tick, waitForCall } from './fake-page.mjs';

const calls = installPage();
const base = '../../src/folio_ocr/web/static/';
const { getState, setState } = await import(`${base}store.js`);
const { onImageError } = await import(`${base}images.js`);
const { markRemoved } = await import(`${base}session.js`);

const BUSY = 'folio is busy reading other files; try again shortly';
const IMAGE_URL = '/api/documents/x/pages/1/image';
let counter = 0;
const freshId = () => { counter += 1; return `doc-images-${counter}`; };
const flag = (id, n) => getState().imageFailed[`${id}:${n}`];
const nextCall = () => waitForCall(calls, () => true);
const secondCall = (first) => waitForCall(calls, (c) => c !== first);

beforeEach(() => {
  calls.length = 0;
  calls.statusFor = () => 200;
  setState({ ...blankState, imageFailed: {} });
});

test('the failed flag is set at once, before any reply', async () => {
  const id = freshId();
  const done = onImageError(id, 1, IMAGE_URL);
  assert.equal(flag(id, 1), true);
  const call = await nextCall();
  assert.equal(flag(id, 1), true);
  call.answer({});
  await done;
});

test('exactly one GET to the image url is sent', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  const done = onImageError(id, 1, IMAGE_URL);
  const call = await nextCall();
  call.answer({ error: BUSY });
  await done;
  await tick();
  assert.equal(calls.length, 1);
  assert.equal(call.url, IMAGE_URL);
  assert.equal(call.init.method, 'GET');
});

test('a busy 503 reply marks the image as busy', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  const done = onImageError(id, 1, IMAGE_URL);
  (await nextCall()).answer({ error: BUSY });
  await done;
  assert.equal(flag(id, 1), 'busy');
});

const stays = [
  ['a 200 reply', 200, {}],
  ['a 503 with another line', 503, { error: 'Files missing' }],
  ['a 503 with the busy line in capitals', 503, { error: BUSY.toUpperCase() }],
  ['a 503 with the busy line and more', 503, { error: `${BUSY} now` }],
  ['a 503 without a line', 503, {}],
  ['a 400 with the busy line', 400, { error: BUSY }],
  ['a 404 with the busy line', 404, { error: BUSY }],
];
for (const [name, status, body] of stays) {
  test(`${name} leaves the image as failed`, async () => {
    const id = freshId();
    calls.statusFor = () => status;
    const done = onImageError(id, 1, IMAGE_URL);
    (await nextCall()).answer(body);
    await done;
    assert.equal(flag(id, 1), true);
  });
}

test('a 503 with an unreadable body leaves the image as failed', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  const done = onImageError(id, 1, IMAGE_URL);
  (await nextCall()).failBody(new SyntaxError('bad json'));
  await done;
  assert.equal(flag(id, 1), true);
});

test('a network failure leaves the image as failed', async () => {
  const id = freshId();
  const fake = window.fetch;
  window.fetch = () => Promise.reject(new TypeError('offline'));
  try {
    await onImageError(id, 1, IMAGE_URL);
  } finally {
    window.fetch = fake;
  }
  assert.equal(flag(id, 1), true);
});

test('a busy reply does not bring back a flag a finished read removed', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  const done = onImageError(id, 1, IMAGE_URL);
  const call = await nextCall();
  setState({ imageFailed: {} });
  call.answer({ error: BUSY });
  await done;
  assert.equal(flag(id, 1), undefined);
});

test('a busy body that arrives after the flag was removed changes nothing', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  const done = onImageError(id, 1, IMAGE_URL);
  const call = await nextCall();
  await tick();
  setState({ imageFailed: {} });
  call.answer({ error: BUSY });
  await done;
  assert.equal(flag(id, 1), undefined);
});

test('the reply to an older ask changes nothing while a newer ask is pending', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  const older = onImageError(id, 1, IMAGE_URL);
  const first = await nextCall();
  const newer = onImageError(id, 1, IMAGE_URL);
  const second = await secondCall(first);
  first.answer({ error: BUSY });
  await older;
  assert.equal(flag(id, 1), true);
  second.answer({ error: BUSY });
  await newer;
  assert.equal(flag(id, 1), 'busy');
});

test('the newer ask decides when it is not busy but the older one was', async () => {
  const id = freshId();
  calls.statusFor = () => (calls.length === 0 ? 503 : 200);
  const older = onImageError(id, 1, IMAGE_URL);
  const first = await nextCall();
  const newer = onImageError(id, 1, IMAGE_URL);
  const second = await secondCall(first);
  second.answer({});
  await newer;
  first.answer({ error: BUSY });
  await older;
  assert.equal(flag(id, 1), true);
});

test('a busy body that arrives after a newer ask was made changes nothing', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  const older = onImageError(id, 1, IMAGE_URL);
  const first = await nextCall();
  await tick();
  const newer = onImageError(id, 1, IMAGE_URL);
  const second = await secondCall(first);
  first.answer({ error: BUSY });
  await older;
  assert.equal(flag(id, 1), true);
  second.answer({});
  await newer;
});

test('a busy reply for a removed document changes nothing', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  const done = onImageError(id, 1, IMAGE_URL);
  const call = await nextCall();
  markRemoved(id);
  call.answer({ error: BUSY });
  await done;
  assert.equal(flag(id, 1), true);
});

test('a busy reply for page 1 does not touch page 2', async () => {
  const id = freshId();
  calls.statusFor = () => 503;
  setState({ imageFailed: { [`${id}:2`]: true } });
  const done = onImageError(id, 1, IMAGE_URL);
  (await nextCall()).answer({ error: BUSY });
  await done;
  assert.equal(flag(id, 1), 'busy');
  assert.equal(flag(id, 2), true);
});
