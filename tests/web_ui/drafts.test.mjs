import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { blankState, elements, installPage, makeDoc, makeLine, stubElement, tick } from './fake-page.mjs';

const calls = installPage();
const base = '../../src/folio_ocr/web/static/';
const { getState, setState } = await import(`${base}store.js`);
const lines = await import(`${base}lines.js`);
const { goPage, goStart } = await import(`${base}docs.js`);
const { msg } = await import(`${base}copy.js`);

const BLOCKED = ['Save (Enter) or cancel (Esc) your change to line 1 first.'];
const page = (index, status, lineList) => ({ index, status, width: 10, height: 10, lines: lineList });

let n = 0;
let id;
/** Open a two-page document with line 1 selected and an edit box holding `value`. */
function open({ stored = 'old', value = 'changed', pages } = {}) {
  n += 1;
  id = `dr${n}`;
  const base = makeDoc(id, [makeLine('p1-l1', { final_text: stored }), makeLine('p1-l2')]);
  const doc = { ...base, pages: pages ?? [base.pages[0], page(2, 'read', [makeLine('p2-l1')])] };
  setState({
    docs: { [id]: doc }, screen: 'doc', docId: id, page: 1, lineId: 'p1-l1',
    draft: { docId: id, lineId: 'p1-l1', value },
  });
}

function assertBlocked(expectedDraft = 'changed') {
  const s = getState();
  assert.equal(s.draft.value, expectedDraft);
  assert.equal(s.draft.lineId, 'p1-l1');
  assert.deepEqual(s.alert, [msg.unsavedDraft(1)]);
  assert.deepEqual(s.alert, BLOCKED);
  assert.equal(elements.editor.focused, 1);
}

beforeEach(() => {
  calls.length = 0;
  elements.lines = stubElement();
  elements.editor = stubElement();
  elements['start-title'] = stubElement();
  setState(blankState);
});

test('selecting another line is blocked while the edit box holds a changed text', () => {
  open();
  lines.selectLine('p1-l2');
  assert.equal(getState().lineId, 'p1-l1');
  assertBlocked();
  assert.equal(calls.length, 0);
});

test('moving to the next line is blocked while the edit box holds a changed text', () => {
  open();
  lines.next(1);
  assert.equal(getState().page, 1);
  assert.equal(getState().lineId, 'p1-l1');
  assertBlocked();
});

test('going to another page is blocked while the edit box holds a changed text', () => {
  open();
  goPage(2);
  assert.equal(getState().page, 1);
  assert.equal(getState().lineId, 'p1-l1');
  assertBlocked();
});

test('going to the start screen is blocked while the edit box holds a changed text', async () => {
  open();
  goStart();
  await tick();
  assert.equal(getState().screen, 'doc');
  assert.equal(getState().docId, id);
  assertBlocked();
  assert.equal(calls.filter((c) => c.url === '/api/documents' && (c.init?.method ?? 'GET') === 'GET').length, 0);
});

test('confirming is blocked while the edit box holds a changed text', () => {
  open();
  lines.confirm();
  assert.equal(calls.filter((c) => c.init?.method === 'PUT').length, 0);
  assertBlocked();
});

test('an edit box equal to the stored text after CRLF to LF does not block moving', () => {
  open({ stored: 'a\r\nb', value: 'a\nb' });
  lines.selectLine('p1-l2');
  assert.equal(getState().lineId, 'p1-l2');
  assert.equal(getState().draft, null);
  assert.equal(getState().alert, null);
});

test('cancelling an edit drops the changed text, keeps the stored text and focuses the list', () => {
  open();
  lines.cancelEdit();
  assert.equal(getState().draft, null);
  assert.equal(elements.lines.focused, 1);
  assert.equal(getState().docs[id].pages[0].lines[0].final_text, 'old');
  assert.equal(calls.length, 0);
});

const done = (i) => makeLine(`p1-l${i}`, { status: i === 1 ? 'confirmed' : 'auto_accepted' });
const allChecked = (extraPages = []) => open({
  stored: 'x',
  value: 'x',
  pages: [page(1, 'read', [done(1), done(2)]), ...extraPages],
});

test('with every line on read pages checked, next says no lines are left and stays put', () => {
  allChecked();
  lines.next(1);
  assert.equal(getState().status, 'No lines left to check on read pages.');
  assert.equal(getState().lineId, 'p1-l1');
});

test('the no-lines-left message counts one unread page', () => {
  allChecked([page(2, 'unread', [])]);
  lines.next(1);
  assert.equal(getState().status, 'No lines left to check on read pages. 1 page isn’t read yet.');
});

test('the no-lines-left message counts two unread pages', () => {
  allChecked([page(2, 'unread', []), page(3, 'unread', [])]);
  lines.next(1);
  assert.equal(getState().status, 'No lines left to check on read pages. 2 pages aren’t read yet.');
});
