import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { blankState, elements, installPage, makeDoc, makeLine, stubElement, tick } from './fake-page.mjs';

const calls = installPage();
const base = '../../src/folio_ocr/web/static/';
const { getState, setState } = await import(`${base}store.js`);
const lines = await import(`${base}lines.js`);
const { forget, openDoc } = await import(`${base}docs.js`);
const { msg } = await import(`${base}copy.js`);

let n = 0;
let id;
/** Open a document with line 1 selected holding `text`; line 2 also needs checking. */
function open(text, draft = null) {
  n += 1;
  id = `ln${n}`;
  const doc = makeDoc(id, [makeLine('p1-l1', { final_text: text }), makeLine('p1-l2')]);
  setState({
    docs: { [id]: doc },
    screen: 'doc',
    docId: id,
    page: 1,
    lineId: 'p1-l1',
    draft: draft && { docId: id, lineId: 'p1-l1', value: draft },
  });
}
const reply = (text, status = 'edited') => makeLine('p1-l1', { status, final_text: text });

beforeEach(() => {
  calls.length = 0;
  elements.lines = stubElement();
  elements.editor = stubElement();
  setState(blankState);
});

test('confirm sends the stored text byte for byte with the JSON header, even with CRLF', async () => {
  open('a\r\nb');
  lines.confirm();
  const { url, init } = calls[0];
  assert.equal(url, `/api/documents/${id}/lines/p1-l1`);
  assert.equal(init.method, 'PUT');
  assert.deepEqual(init.headers, { 'Content-Type': 'application/json' });
  assert.equal(init.body, JSON.stringify({ final_text: 'a\r\nb' }));
  calls[0].answer(reply('a\r\nb', 'confirmed'));
  await tick();
  assert.equal(getState().status, 'Line 1 confirmed.');
  assert.equal(getState().lineId, 'p1-l2');
});

test('save with a box equal to the stored text after CRLF to LF sends the stored text', () => {
  open('a\r\nb', 'a\nb');
  lines.save();
  assert.equal(calls[0].init.body, JSON.stringify({ final_text: 'a\r\nb' }));
});

test('save of new text reports saved and moves to the next line to check', async () => {
  open('old', 'new');
  lines.save();
  assert.equal(calls[0].init.body, JSON.stringify({ final_text: 'new' }));
  calls[0].answer(reply('new'));
  await tick();
  assert.equal(getState().status, 'Line 1 saved.');
  assert.equal(getState().lineId, 'p1-l2');
  assert.equal(getState().draft, null);
  assert.equal(getState().docs[id].pages[0].lines[0].final_text, 'new');
});

test('a document forgotten while the save is pending is not changed', async () => {
  open('old', 'new');
  lines.save();
  await tick();
  forget(id);
  calls[0].answer(reply('new'));
  await tick();
  assert.equal(getState().docs[id], undefined);
  assert.equal(getState().status, '');
});

test('text typed while a save is in flight stays in the box', async () => {
  open('old', 'new');
  lines.save();
  lines.draftInput('newer');
  calls[0].answer(reply('new'));
  await tick();
  assert.equal(getState().draft.value, 'newer');
});

test('a 400 with a server line shows it and keeps the draft', async () => {
  open('old', 'new');
  lines.save();
  calls[0].status = 400;
  calls[0].answer({ error: 'x' });
  await tick();
  assert.deepEqual(getState().alert, ['Couldn’t save line 1: x. Your text is still in the box.']);
  assert.equal(getState().draft.value, 'new');
});

test('editing again while a save is in flight keeps the new text, says saved and leaves focus alone', async () => {
  open('old', 'new');
  lines.save();
  lines.draftInput('newer');
  calls[0].answer(reply('new'));
  await tick();
  assert.equal(getState().status, 'Line 1 saved.');
  assert.equal(getState().alert, null);
  assert.equal(getState().draft.value, 'newer');
  assert.equal(getState().lineId, 'p1-l1');
  assert.equal(elements.lines.focused, 0);
});

test('an edit box opened in one document is dropped on opening another', async () => {
  open('a');
  const a = id;
  lines.openEditor();
  assert.equal(getState().draft.docId, a);
  const opening = openDoc('ln-other');
  calls[0].answer(makeDoc('ln-other', [makeLine('p1-l1', { final_text: 'b' })]));
  assert.equal(await opening, true);
  assert.equal(getState().draft, null);
  lines.openEditor();
  assert.equal(getState().draft.docId, 'ln-other');
  assert.equal(getState().draft.value, 'b');
});

test('typing while another document is being fetched keeps the screen and shows the unsaved-text alert', async () => {
  open('a');
  const a = id;
  lines.openEditor();
  const opening = openDoc('ln-typed');
  lines.draftInput('typed');
  calls[0].answer(makeDoc('ln-typed', [makeLine('p1-l1')]));
  assert.equal(await opening, false);
  assert.equal(getState().docId, a);
  assert.equal(getState().screen, 'doc');
  assert.deepEqual(getState().alert, [msg.unsavedDraft(1)]);
});

test('confirm with no edit box goes to the next line to check and focuses the list', async () => {
  open('x');
  lines.confirm();
  calls[0].answer(reply('x', 'confirmed'));
  await tick();
  assert.equal(getState().lineId, 'p1-l2');
  assert.equal(elements.lines.focused, 1);
});
