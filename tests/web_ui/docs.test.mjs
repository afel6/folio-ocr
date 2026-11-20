import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { blankState, installPage, makeDoc, makeLine, tick } from './fake-page.mjs';

const calls = installPage();
const base = '../../src/folio_ocr/web/static/';
const { getState, setState } = await import(`${base}store.js`);
const docs = await import(`${base}docs.js`);
const { STALE } = await import(`${base}api.js`);
const { msg } = await import(`${base}copy.js`);

beforeEach(() => {
  calls.length = 0;
  calls.holdHeaders = false;
  setState(blankState);
});

const isOpen = (id) => (c) => c.init.method === 'GET' && c.url === `/api/documents/${id}/export?format=json`;
const held = (id) => ({ [id]: makeDoc(id, [makeLine('p1-l1')]) });

test('loadHealth ignores an older reply that arrives after a newer call', async () => {
  const a = docs.loadHealth();
  const b = docs.loadHealth();
  calls[1].answer({ name: 'new' });
  await b;
  calls[0].answer({ name: 'old' });
  await a;
  assert.deepEqual(getState().health, { name: 'new' });
});

test('loadList returns STALE when a newer call was made', async () => {
  const a = docs.loadList();
  const b = docs.loadList();
  calls[0].answer([{ document_id: 'x' }]);
  assert.equal(await a, STALE);
  calls[1].answer([{ document_id: 'y' }]);
  assert.deepEqual(await b, [{ document_id: 'y' }]);
  assert.deepEqual(getState().list, [{ document_id: 'y' }]);
});

test('openDoc superseded by a newer openDoc returns false and keeps the screen', async () => {
  const a = docs.openDoc('sup-a');
  const b = docs.openDoc('sup-b');
  calls[0].answer(makeDoc('sup-a', [makeLine('p1-l1')]));
  assert.equal(await a, false);
  assert.equal(getState().screen, 'start');
  calls[1].answer(makeDoc('sup-b', [makeLine('p1-l1')]));
  assert.equal(await b, true);
  assert.equal(getState().docId, 'sup-b');
  assert.equal(getState().screen, 'doc');
});

test('openDoc with refetch bumps the image of a page that went from unread to read', async () => {
  const before = makeDoc('bump', [], { pages: [{ index: 1, status: 'unread', width: 10, height: 10, lines: [] }] });
  setState({ docs: { bump: before } });
  const opened = docs.openDoc('bump', { refetch: true });
  calls[0].answer(makeDoc('bump', [makeLine('p1-l1')]));
  assert.equal(await opened, true);
  assert.equal(getState().readCounts['bump:1'], 1);
});

test('refreshEngine drops an older reply once a newer one was asked', async () => {
  setState({ docs: { eng: makeDoc('eng', []) } });
  const a = docs.refreshEngine('eng');
  const b = docs.refreshEngine('eng');
  calls[1].answer({ engine: { reader: 'new' } });
  await b;
  calls[0].answer({ engine: { reader: 'old' } });
  await a;
  assert.equal(getState().docs.eng.engine.reader, 'new');
});

test('refreshEngine changes nothing for a document forgotten while it was pending', async () => {
  setState({ docs: { gone: makeDoc('gone', []) } });
  const pending = docs.refreshEngine('gone');
  await tick();
  docs.forget('gone');
  calls[0].answer({ engine: { reader: 'x' } });
  await pending;
  assert.equal(getState().docs.gone, undefined);
});

test('refreshEngine shows no alert when a forgotten document answers 404', async () => {
  setState({ docs: { gone404: makeDoc('gone404', []) } });
  const pending = docs.refreshEngine('gone404');
  calls[0].status = 404;
  await tick();
  docs.forget('gone404');
  calls[0].answer({ error: 'no such document' });
  await pending;
  assert.equal(getState().alert, null);
});

test('a 404 on openDoc forgets the document and sets the docGone alert', async () => {
  setState({ list: [{ document_id: 'nf', filename: 'nf.png' }] });
  const opened = docs.openDoc('nf');
  calls[0].status = 404;
  calls[0].answer({ error: 'no such document' });
  assert.equal(await opened, false);
  await tick();
  assert.deepEqual(getState().alert, [msg.docGone('nf.png')]);
  assert.deepEqual(getState().list, []);
  assert.equal(getState().docs.nf, undefined);
});

test('a slow open whose body arrives late does not override a later open of a document already held', async () => {
  setState({ docs: held('race-held') });
  const slow = docs.openDoc('race-slow');
  await tick();
  assert.equal(await docs.openDoc('race-held'), true);
  calls.find(isOpen('race-slow')).answer(makeDoc('race-slow', [makeLine('p1-l1')]));
  assert.equal(await slow, false);
  assert.equal(getState().screen, 'doc');
  assert.equal(getState().docId, 'race-held');
  assert.equal(getState().docs['race-slow'], undefined);
});

test('a slow open whose headers arrive late does not override a later open of a document already held', async () => {
  calls.holdHeaders = true;
  setState({ docs: held('race-held2') });
  const slow = docs.openDoc('race-slow2');
  assert.equal(await docs.openDoc('race-held2'), true);
  const call = calls.find(isOpen('race-slow2'));
  call.sendHeaders();
  call.answer(makeDoc('race-slow2', [makeLine('p1-l1')]));
  assert.equal(await slow, false);
  assert.equal(getState().docId, 'race-held2');
});

test('a slow open does not override going to Start', async () => {
  const slow = docs.openDoc('race-start');
  await tick();
  const going = docs.goStart();
  calls.find(isOpen('race-start')).answer(makeDoc('race-start', [makeLine('p1-l1')]));
  assert.equal(await slow, false);
  calls.find((c) => c.url === '/api/documents').answer([]);
  await going;
  assert.equal(getState().screen, 'start');
  assert.equal(getState().docId, null);
});

test('a slow open that fails after a later move shows no alert', async () => {
  setState({ docs: held('race-held3') });
  const slow = docs.openDoc('race-fail');
  await tick();
  await docs.openDoc('race-held3');
  calls.find(isOpen('race-fail')).failBody(new TypeError('connection reset'));
  assert.equal(await slow, false);
  assert.equal(getState().alert, null);
  assert.equal(getState().docId, 'race-held3');
});

test('a move blocked by a changed edit box does not drop the open on its way', async () => {
  const doc = held('race-draft');
  setState({
    docs: doc, screen: 'doc', docId: 'race-draft', page: 1, lineId: 'p1-l1',
    draft: { docId: 'race-draft', lineId: 'p1-l1', value: 'changed' },
  });
  const refresh = docs.openDoc('race-draft', { refetch: true });
  await docs.goStart();
  assert.equal(getState().screen, 'doc');
  calls.find(isOpen('race-draft')).answer(makeDoc('race-draft', [makeLine('p1-l1', { final_text: 'new' })]));
  assert.equal(await refresh, true);
  assert.equal(getState().docs['race-draft'].pages[0].lines[0].final_text, 'new');
});

const pageWith = (index, lines) => ({ index, status: 'read', width: 10, height: 10, lines });

test('a refresh answered after the user went to page 2 shows page 2', async () => {
  const doc = makeDoc('keep-page', [makeLine('p1-l1')]);
  const twoPages = { ...doc, pages: [doc.pages[0], pageWith(2, [makeLine('p2-l1'), makeLine('p2-l2')])] };
  setState({ docs: { 'keep-page': twoPages }, screen: 'doc', docId: 'keep-page', page: 1, lineId: 'p1-l1' });
  docs.refreshDoc();
  docs.goPage(2);
  calls.find(isOpen('keep-page')).answer(twoPages);
  await tick();
  assert.equal(getState().page, 2);
  assert.equal(getState().lineId, 'p2-l1');
});

test('a refresh answered after the user selected and edited another line keeps that line and its edit box', async () => {
  const doc = makeDoc('keep-line', [makeLine('p1-l1'), makeLine('p1-l2')]);
  setState({ docs: { 'keep-line': doc }, screen: 'doc', docId: 'keep-line', page: 1, lineId: 'p1-l1' });
  docs.refreshDoc();
  const draft = { docId: 'keep-line', lineId: 'p1-l2', value: 'x' };
  setState({ lineId: 'p1-l2', draft });
  calls.find(isOpen('keep-line')).answer({ ...doc });
  await tick();
  assert.equal(getState().lineId, 'p1-l2');
  assert.deepEqual(getState().draft, draft);
});
