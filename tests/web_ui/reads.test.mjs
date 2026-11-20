import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { blankState, elements, installPage, makeDoc, makeLine, stubElement, tick, waitForCall } from './fake-page.mjs';

const calls = installPage();
const base = '../../src/folio_ocr/web/static/';
const { getState, setState } = await import(`${base}store.js`);
const { readPage, stopReading } = await import(`${base}reads.js`);
const { recordSaved } = await import(`${base}session.js`);

const is = (method, re) => (c) => c.init.method === method && re.test(c.url);
const readCall = is('POST', /\/read$/);
const healthGet = is('GET', /^\/api\/health$/);
const exportGet = is('GET', /export\?format=json$/);

let counter = 0;
const freshId = () => { counter += 1; return `doc-reads-${counter}`; };

beforeEach(() => {
  calls.length = 0;
  calls.statusFor = () => 200;
  elements.editor = stubElement();
  elements.lines = stubElement();
  stopReading();
  setState({ ...blankState, imageFailed: {}, queue: [], inFlight: null, paused: false });
});

const waitFor = (match) => waitForCall(calls, match);

/** Answer every health and export request that the read has sent. */
async function answerRefreshes(id) {
  for (const [match, body] of [
    [healthGet, { model: { configured: false } }],
    [exportGet, { engine: makeDoc(id, []).engine }],
  ]) {
    const call = calls.find((c) => !c.answered && match(c));
    if (call) call.answer(body);
  }
  await tick();
}

/** Send a read of page 1 and answer it with `body` (a page), then the follow-up requests. */
async function readAndAnswer(id, body, status = 200) {
  calls.statusFor = (c) => (readCall(c) ? status : 200);
  readPage(id, 1);
  const call = await waitFor(readCall);
  call.answer(body);
  await tick();
  await tick();
  await answerRefreshes(id);
}

const replyPage = (lines, extra = {}) => ({ index: 1, status: 'read', width: 10, height: 10, lines, ...extra });
const finalOf = (id, lineId) => getState().docs[id].pages[0].lines.find((l) => l.id === lineId).final_text;
const withDoc = (id, lines, extra = {}) => setState({ docs: { [id]: makeDoc(id, lines) }, ...extra });

test('a line saved after the read was sent keeps the saved text', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1', { final_text: 'old' })]);
  readPage(id, 1);
  const call = await waitFor(readCall);
  recordSaved(id, makeLine('p1-l1', { status: 'edited', final_text: 'mine' }));
  call.answer(replyPage([makeLine('p1-l1', { final_text: 'reread' })]));
  await tick();
  await tick();
  await answerRefreshes(id);
  assert.equal(finalOf(id, 'p1-l1'), 'mine');
});

test('a line saved before the read was sent takes the re-read text', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1', { final_text: 'old' })]);
  recordSaved(id, makeLine('p1-l1', { status: 'edited', final_text: 'mine' }));
  await readAndAnswer(id, replyPage([makeLine('p1-l1', { final_text: 'reread' })]));
  assert.equal(finalOf(id, 'p1-l1'), 'reread');
});

test('a finished read adds one to the count for that page', async () => {
  const first = freshId();
  withDoc(first, [makeLine('p1-l1')]);
  assert.equal(getState().readCounts[`${first}:1`], undefined);
  await readAndAnswer(first, replyPage([makeLine('p1-l1')]));
  assert.equal(getState().readCounts[`${first}:1`], 1);

  const second = freshId();
  withDoc(second, [makeLine('p1-l1')], { readCounts: { [`${second}:1`]: 2 } });
  await readAndAnswer(second, replyPage([makeLine('p1-l1')]));
  assert.equal(getState().readCounts[`${second}:1`], 3);
});

test('a finished read clears the image failure of that page only', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1')], { imageFailed: { [`${id}:1`]: true, [`${id}:2`]: true } });
  await readAndAnswer(id, replyPage([makeLine('p1-l1')]));
  assert.deepEqual(getState().imageFailed, { [`${id}:2`]: true });
});

test('the selected line moves to the first line to check when it is not in the new reading', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l9')], { docId: id, page: 1, lineId: 'p1-l9' });
  const reply = replyPage([makeLine('p1-l1', { status: 'auto_accepted' }), makeLine('p1-l2')]);
  await readAndAnswer(id, reply);
  assert.equal(getState().lineId, 'p1-l2');
});

test('the selected line stays when it is still in the new reading', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1')], { docId: id, page: 1, lineId: 'p1-l1' });
  await readAndAnswer(id, replyPage([makeLine('p1-l2'), makeLine('p1-l1')]));
  assert.equal(getState().lineId, 'p1-l1');
});

test('the selected line is unchanged when another page is shown', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l9')], { docId: id, page: 2, lineId: 'p1-l9' });
  await readAndAnswer(id, replyPage([makeLine('p1-l1')]));
  assert.equal(getState().lineId, 'p1-l9');
});

const twoLines = () => [makeLine('p1-l1', { status: 'auto_accepted' }), makeLine('p1-l2')];

test('the status after a read counts lines and lines to check', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1')], { docId: id, page: 1, lineId: 'p1-l1' });
  await readAndAnswer(id, replyPage(twoLines()));
  assert.equal(getState().status, 'Page 1 read: 2 lines, 1 to check.');
});

test('the status after a read points to the notes when the page has unread regions', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1')], { docId: id, page: 1, lineId: 'p1-l1' });
  await readAndAnswer(id, replyPage(twoLines(), { unread_regions: [[0, 0, 1, 1]] }));
  assert.equal(getState().status, 'Page 1 read: 2 lines, 1 to check. See the notes about this page.');
});

test('the status is unchanged when another document is shown', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1')], { docId: 'other-doc', status: 'untouched' });
  await readAndAnswer(id, replyPage(twoLines()));
  assert.equal(getState().status, 'untouched');
});

test('health is fetched again after a read that worked', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1')]);
  await readAndAnswer(id, replyPage([makeLine('p1-l1')]));
  assert.equal(calls.filter(healthGet).length, 1);
});

test('health is fetched again after a read that failed, and the page stays as it was', async () => {
  const id = freshId();
  withDoc(id, [makeLine('p1-l1', { final_text: 'old' })]);
  const before = getState().docs[id];
  await readAndAnswer(id, { error: 'x' }, 502);
  assert.equal(calls.filter(healthGet).length, 1);
  assert.deepEqual(getState().docs[id], before);
});
