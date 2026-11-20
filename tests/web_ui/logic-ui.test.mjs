import test from 'node:test';
import assert from 'node:assert/strict';
import * as L from '../../src/folio_ocr/web/static/logic.js';

const eq = assert.deepEqual;

test('moveTarget steps, pages, clamps and jumps', () => {
  assert.equal(L.moveTarget('ArrowDown', 2, 5), 3);
  assert.equal(L.moveTarget('ArrowUp', 2, 5), 1);
  assert.equal(L.moveTarget('ArrowDown', 4, 5), 4);
  assert.equal(L.moveTarget('ArrowUp', 0, 5), 0);
  assert.equal(L.moveTarget('PageDown', 2, 50), 12);
  assert.equal(L.moveTarget('PageUp', 5, 50), 0);
  assert.equal(L.moveTarget('PageDown', 45, 50), 49);
  assert.equal(L.moveTarget('Home', 3, 5), 0);
  assert.equal(L.moveTarget('End', 1, 5), 4);
  assert.equal(L.moveTarget('x', 1, 5), null);
});

test('closeupBox pads by 8 and clamps to the page', () => {
  const page = { width: 100, height: 50 };
  eq(L.closeupBox(page, { box: [20, 10, 60, 30] }), [12, 2, 56, 36]);
  eq(L.closeupBox(page, { box: [0, 0, 100, 50] }), [0, 0, 100, 50]);
  eq(L.closeupBox(page, { box: [95, 45, 99, 49] }), [87, 37, 13, 13]);
});

test('closeupBox falls back to the whole page for an empty box', () => {
  eq(L.closeupBox({ width: 100, height: 50 }, { box: [200, 10, 210, 20] }), [0, 0, 100, 50]);
});

test('isRemoteHost', () => {
  assert.equal(L.isRemoteHost('localhost'), false);
  assert.equal(L.isRemoteHost('127.0.0.1'), false);
  assert.equal(L.isRemoteHost('[::1]'), false);
  assert.equal(L.isRemoteHost('example.org'), true);
  assert.equal(L.isRemoteHost('192.168.1.2'), true);
});

test('countText', () => {
  const pg = (status, lines = []) => ({ status, lines });
  const ln = (status) => ({ status });
  assert.equal(L.countText({ unresolved_count: 3, pages: [] }, null), '3 to check');
  assert.equal(L.countText({ unresolved_count: 0, pages: [{ status: 'read' }] }, null), 'done');
  assert.equal(L.countText({ unresolved_count: 0, pages: [{ status: 'unread' }] }, null), '');
  const doc = { pages: [pg('read', [ln('needs_review'), ln('confirmed')])] };
  assert.equal(L.countText({ unresolved_count: 0, pages: [] }, doc), '1 to check');
  assert.equal(L.countText({ unresolved_count: 9, pages: [] }, { pages: [pg('read')] }), 'done');
});

test('countText counts pages with unread areas for loaded and unloaded documents alike, never as lines', () => {
  const partly = { status: 'read', lines: [{ status: 'confirmed' }], unread_regions: [[0, 0, 1, 1]] };
  const unloaded = { unresolved_count: 1, pages: [{ status: 'read' }] };
  assert.equal(L.countText(unloaded, null), '1 to check');
  assert.equal(L.countText(unloaded, { pages: [partly] }), '1 to check');
  const both = { pages: [partly, { status: 'read', lines: [{ status: 'needs_review' }] }] };
  assert.equal(L.countText({ unresolved_count: 0, pages: [] }, both), '2 to check');
  assert.doesNotMatch(L.countText({ unresolved_count: 4, pages: [] }, null), /line/);
});

test('readingPages lists reading pages and the page in flight in this document', () => {
  const doc = { pages: [{ index: 1, status: 'read' }, { index: 2, status: 'reading' }, { index: 3, status: 'unread' }] };
  eq(L.readingPages({ docId: 'd', inFlight: null }, doc), [2]);
  eq(L.readingPages({ docId: 'd', inFlight: { docId: 'd', page: 3 } }, doc), [2, 3]);
  eq(L.readingPages({ docId: 'd', inFlight: { docId: 'other', page: 3 } }, doc), [2]);
});

test('nextDocId', () => {
  const list = [{ document_id: 'a' }, { document_id: 'b' }, { document_id: 'c' }];
  assert.equal(L.nextDocId(list, 'a'), 'b');
  assert.equal(L.nextDocId(list, 'c'), 'a');
  assert.equal(L.nextDocId([{ document_id: 'a' }], 'a'), null);
});

test('clampPage', () => {
  const doc = { pages: [1, 2, 3] };
  assert.equal(L.clampPage(doc, '2'), 2);
  assert.equal(L.clampPage(doc, 3), 3);
  assert.equal(L.clampPage(doc, 4), 1);
  assert.equal(L.clampPage(doc, 0), 1);
  assert.equal(L.clampPage(doc, 'x'), 1);
  assert.equal(L.clampPage(doc, null), 1);
});

test('readButtonLabel', () => {
  const idle = { queued: false, readingHere: false };
  assert.equal(L.readButtonLabel({ status: 'unread' }, idle), 'Read this page');
  assert.equal(L.readButtonLabel({ status: 'read' }, idle), 'Read again');
  assert.equal(L.readButtonLabel({ status: 'failed' }, idle), 'Read again');
  assert.equal(L.readButtonLabel({ status: 'reading' }, idle), null);
  assert.equal(L.readButtonLabel({ status: 'unread' }, { queued: true, readingHere: false }), null);
  assert.equal(L.readButtonLabel({ status: 'read' }, { queued: false, readingHere: true }), null);
});

test('isQueued is true for a waiting page, false when paused or for another document or page', () => {
  const queue = [{ docId: 'd', page: 2 }];
  assert.equal(L.isQueued({ paused: false, queue }, 'd', 2), true);
  assert.equal(L.isQueued({ paused: true, queue }, 'd', 2), false);
  assert.equal(L.isQueued({ paused: false, queue }, 'other', 2), false);
  assert.equal(L.isQueued({ paused: false, queue }, 'd', 3), false);
});

const ids = ['a', 'b', 'c', 'd', 'e'].map((id) => ({ id }));
const press = (key, extra = {}) => ({ key, ...extra });

test('listKeyAction moves the selection with focus on the list', () => {
  assert.deepEqual(L.listKeyAction(press('ArrowDown'), ids, 'b'), ['selectLine', 'c', { focusList: true }]);
  assert.deepEqual(L.listKeyAction(press('ArrowUp'), ids, 'b'), ['selectLine', 'a', { focusList: true }]);
  assert.deepEqual(L.listKeyAction(press('Home'), ids, 'c'), ['selectLine', 'a', { focusList: true }]);
  assert.deepEqual(L.listKeyAction(press('End'), ids, 'c'), ['selectLine', 'e', { focusList: true }]);
  assert.deepEqual(L.listKeyAction(press('PageDown'), ids, 'a'), ['selectLine', 'e', { focusList: true }]);
  assert.deepEqual(L.listKeyAction(press('PageUp'), ids, 'c'), ['selectLine', 'a', { focusList: true }]);
});

test('listKeyAction clamps at the ends and starts at the first line when none is selected', () => {
  assert.deepEqual(L.listKeyAction(press('ArrowDown'), ids, 'e'), ['selectLine', 'e', { focusList: true }]);
  assert.deepEqual(L.listKeyAction(press('ArrowUp'), ids, 'a'), ['selectLine', 'a', { focusList: true }]);
  assert.deepEqual(L.listKeyAction(press('ArrowDown'), ids, 'zzz'), ['selectLine', 'a', { focusList: true }]);
  assert.deepEqual(L.listKeyAction(press('ArrowDown'), ids, null), ['selectLine', 'a', { focusList: true }]);
});

test('listKeyAction maps next, confirm and edit keys', () => {
  assert.deepEqual(L.listKeyAction(press('n'), ids, 'a'), ['next', 1]);
  assert.deepEqual(L.listKeyAction(press('N', { shiftKey: true }), ids, 'a'), ['next', -1]);
  assert.deepEqual(L.listKeyAction(press('c'), ids, 'a'), ['confirm']);
  assert.deepEqual(L.listKeyAction(press('C'), ids, 'a'), ['confirm']);
  assert.deepEqual(L.listKeyAction(press('e'), ids, 'a'), ['openEditor']);
  assert.deepEqual(L.listKeyAction(press('E'), ids, 'a'), ['openEditor']);
  assert.deepEqual(L.listKeyAction(press('Enter'), ids, 'a'), ['openEditor']);
});

test('listKeyAction leaves unknown keys and modified keys alone', () => {
  assert.equal(L.listKeyAction(press('x'), ids, 'a'), null);
  for (const mod of ['ctrlKey', 'altKey', 'metaKey']) {
    assert.equal(L.listKeyAction(press('n', { [mod]: true }), ids, 'a'), null);
    assert.equal(L.listKeyAction(press('ArrowDown', { [mod]: true }), ids, 'a'), null);
  }
});

test('listKeyAction on an empty list ignores movement but still maps N, C, E and Enter', () => {
  for (const key of ['ArrowDown', 'ArrowUp', 'Home', 'End', 'PageUp', 'PageDown']) {
    assert.equal(L.listKeyAction(press(key), [], null), null);
  }
  assert.deepEqual(L.listKeyAction(press('n'), [], null), ['next', 1]);
  assert.deepEqual(L.listKeyAction(press('c'), [], null), ['confirm']);
  assert.deepEqual(L.listKeyAction(press('e'), [], null), ['openEditor']);
  assert.deepEqual(L.listKeyAction(press('Enter'), [], null), ['openEditor']);
});

test('editorKeyAction saves on Enter and cancels on Escape', () => {
  assert.deepEqual(L.editorKeyAction(press('Enter')), ['save']);
  assert.equal(L.editorKeyAction(press('Enter', { shiftKey: true })), null);
  assert.equal(L.editorKeyAction(press('Enter', { isComposing: true })), null);
  assert.deepEqual(L.editorKeyAction(press('Escape')), ['cancelEdit']);
  assert.deepEqual(L.editorKeyAction(press('Escape', { shiftKey: true })), ['cancelEdit']);
  assert.equal(L.editorKeyAction(press('a')), null);
});

test('drawParts draws every part after one throws and reports the failure', () => {
  const drawn = [];
  const ok = (name) => ({ render: (state) => drawn.push([name, state]) });
  const bad = { render: () => { throw new Error('x'); } };
  const state = { n: 1 };
  assert.equal(L.drawParts([ok('a'), bad, ok('c')], state), true);
  assert.deepEqual(drawn, [['a', state], ['c', state]]);
});

test('drawParts reports no failure when every part draws', () => {
  assert.equal(L.drawParts([{ render() {} }, { render() {} }], {}), false);
});

test('alertOnDrawFailure is true only for a new failure', () => {
  assert.equal(L.alertOnDrawFailure(true, false), true);
  assert.equal(L.alertOnDrawFailure(true, true), false);
  assert.equal(L.alertOnDrawFailure(false, false), false);
  assert.equal(L.alertOnDrawFailure(false, true), false);
  let before = false;
  const alerts = [true, true, false, true].map((failed) => {
    const alert = L.alertOnDrawFailure(failed, before);
    before = failed;
    return alert;
  });
  assert.deepEqual(alerts, [true, false, false, true]);
});
