import test from 'node:test';
import assert from 'node:assert/strict';
import * as L from '../../src/folio_ocr/web/static/logic.js';

const eq = assert.deepEqual;
const line = (id, status = 'auto_accepted', flags = [], extra = {}) => ({
  id, status, flags, text: 't', final_text: 't', direction: 'ltr', script: 'latin', ...extra,
});
const page = (index, status, lines = [], extra = {}) => ({
  index, status, lines, flags: [], unread_regions: [], invalid_lines: 0, error: null, ...extra,
});
const doc = (pages, extra = {}) => ({ id: 'd', pages, engine: { reader: 'onnxtr', calibration: { tau: 0.5 } }, ...extra });
const readCtx = { queued: false, readingHere: false, imageFailed: false, modelConfigured: false };

test('lineAttrs', () => {
  eq(L.lineAttrs({ direction: 'rtl', script: 'arabic' }), { dir: 'rtl', lang: 'ar' });
  eq(L.lineAttrs({ direction: 'rtl', script: 'mixed' }), { dir: 'rtl', lang: 'ar' });
  eq(L.lineAttrs({ direction: 'ltr', script: 'mixed' }), { dir: 'ltr', lang: 'en' });
  eq(L.lineAttrs({ direction: 'ltr', script: 'latin' }), { dir: 'ltr', lang: 'en' });
  eq(L.lineAttrs({ direction: 'ltr', script: 'unknown' }), { dir: 'ltr', lang: null });
  eq(L.lineAttrs({ direction: 'sideways', script: 'arabic' }), { dir: 'auto', lang: 'ar' });
  eq(L.lineAttrs({ script: 'mixed' }), { dir: 'auto', lang: 'en' });
});

test('lineNumber', () => {
  assert.equal(L.lineNumber({ id: 'p2-l15' }, 0), 15);
  assert.equal(L.lineNumber({ id: 'weird' }, 4), 5);
});

test('counts', () => {
  const d = doc([
    page(1, 'read', [line('p1-l1', 'needs_review'), line('p1-l2'), line('p1-l3', 'needs_review')]),
    page(2, 'unread'), page(3, 'failed'), page(4, 'reading'),
  ]);
  assert.equal(L.toCheck(d.pages[0].lines), 2);
  assert.equal(L.docToCheck(d), 2);
  eq(L.notReadPages(d).map((p) => p.index), [2, 3, 4]);
});

test('nextToCheck', () => {
  const d = doc([
    page(1, 'read', [line('p1-l1', 'needs_review'), line('p1-l2'), line('p1-l3', 'needs_review')]),
    page(2, 'read', [line('p2-l1', 'needs_review')]),
    page(3, 'unread'),
  ]);
  eq(L.nextToCheck(d, 0, 'p1-l1', 1), { page: 0, lineId: 'p1-l3' });
  eq(L.nextToCheck(d, 0, 'p1-l3', 1), { page: 1, lineId: 'p2-l1' });
  eq(L.nextToCheck(d, 1, 'p2-l1', 1), { page: 0, lineId: 'p1-l1' });
  eq(L.nextToCheck(d, 0, 'p1-l1', -1), { page: 1, lineId: 'p2-l1' });
  eq(L.nextToCheck(d, 1, 'p2-l1', -1), { page: 0, lineId: 'p1-l3' });
  eq(L.nextToCheck(d, 0, 'p1-l2', 1), { page: 0, lineId: 'p1-l3' });
  eq(L.nextToCheck(d, 0, 'p1-l2', -1), { page: 0, lineId: 'p1-l1' });
  eq(L.nextToCheck(d, 0, null, 1), { page: 0, lineId: 'p1-l1' });
  const none = doc([page(1, 'read', [line('p1-l1')])]);
  assert.equal(L.nextToCheck(none, 0, 'p1-l1', 1), null);
  const one = doc([page(1, 'read', [line('p1-l1', 'needs_review'), line('p1-l2')])]);
  eq(L.nextToCheck(one, 0, 'p1-l1', 1), { page: 0, lineId: 'p1-l1' });
  eq(L.nextToCheck(one, 0, 'p1-l1', -1), { page: 0, lineId: 'p1-l1' });
});

test('everyLineFlagged', () => {
  assert.equal(L.everyLineFlagged({ engine: { reader: 'onnxtr', calibration: null } }), true);
  assert.equal(L.everyLineFlagged({ engine: { reader: 'onnxtr', calibration: { tau: null } } }), true);
  assert.equal(L.everyLineFlagged({ engine: { reader: 'onnxtr', calibration: { tau: 0.4 } } }), false);
  assert.equal(L.everyLineFlagged({ engine: { reader: 'none', calibration: null } }), false);
});

test('exportSummary', () => {
  const check = [line('p1-l1', 'needs_review'), line('p1-l2', 'needs_review'), line('p1-l3', 'needs_review')];
  const base = doc([page(1, 'read', check)]);
  assert.equal(L.exportSummary(base, []), '3 lines still need checking. They’re exported as read, and the Word file highlights them.');
  assert.equal(L.exportSummary(doc([page(1, 'read', [line('p1-l1', 'needs_review')])]), []),
    '1 line still needs checking. It’s exported as read, and the Word file highlights it.');
  assert.equal(L.exportSummary(doc([page(1, 'read', [line('p1-l1')])]), []), 'No lines left to check.');
  const mixed = doc([
    page(1, 'read', [], { unread_regions: [[0, 0, 1, 1]] }), page(2, 'unread'), page(3, 'failed'), page(4, 'reading'),
  ]);
  assert.equal(L.exportSummary(mixed, [4]),
    'No lines left to check. 2 pages aren’t read, so they’ll be empty in the export. '
    + 'Parts of 1 page weren’t read, so their text isn’t in the export. '
    + 'Page 4 is still being read and won’t be in an export made now.');
  const two = doc([page(1, 'read'), page(2, 'unread'), page(3, 'reading'), page(4, 'reading')]);
  assert.equal(L.exportSummary(two, [3, 4]),
    'No lines left to check. 1 page isn’t read, so it’ll be empty in the export. '
    + 'Pages 3 and 4 are still being read and won’t be in an export made now.');
  assert.equal(L.exportSummary(doc([page(1, 'unread')]), [1]).includes('empty in the export'), false);
});

const health = (over = {}) => ({
  status: 'ok', weights: 'ok', tesseract: 'ok',
  model: { configured: false, reachable: null, name: null, host: null, leaves_machine: null, fit: 'untested', calibrated: 'ok' },
  ...over,
});
const withModel = (m) => health({ model: { ...health().model, configured: true, name: 'qwen', host: 'gpu-box', ...m } });

test('readersState', () => {
  assert.equal(L.readersState(null), 'unknown');
  assert.equal(L.readersState(health(), ['x']), 'unknown');
  assert.equal(L.readersState(health()), 'ready');
  assert.equal(L.readersState(health({ weights: 'uncalibrated', tesseract: 'uncalibrated' })), 'ready');
  assert.equal(L.readersState(health({ weights: 'missing' })), 'check');
  assert.equal(L.readersState(health({ weights: 'bad_data' })), 'check');
  assert.equal(L.readersState(health({ tesseract: 'bad_data' })), 'check');
  assert.equal(L.readersState(health({ tesseract: 'missing' })), 'check');
  assert.equal(L.readersState({ ...withModel({ reachable: true, fit: 'pass' }), tesseract: 'missing' }), 'ready');
  assert.equal(L.readersState(withModel({ reachable: false })), 'check');
  assert.equal(L.readersState(withModel({ reachable: true, fit: 'fail' })), 'check');
  assert.equal(L.readersState(withModel({ reachable: true, fit: 'pass', calibrated: 'uncalibrated' })), 'ready');
});

test('readersRows', () => {
  const rows = L.readersRows(health());
  eq(rows.map((r) => r.name), ['Scan reader', 'Tesseract (Arabic and English)', 'Reading model']);
  eq(rows[0].segments, ['Ready.']);
  eq(rows[2].segments, ['Not set up (optional).']);
  eq(L.readersRows(health({ weights: 'uncalibrated' }))[0].segments, ['Ready. Its confidence check isn’t set up yet, so every line it reads is marked to check.']);
  eq(L.readersRows(health({ weights: 'missing' }))[0].segments, ['Its files are missing. Ask your administrator to run ', { code: 'folio setup' }, '.']);
  eq(L.readersRows(health({ weights: 'bad_data' }))[0].segments, ['Its files can’t be used. Ask your administrator to run ', { code: 'folio doctor' }, '.']);
  eq(L.readersRows(health({ tesseract: 'missing' }))[1].segments, ['Not installed (optional). Without it or a reading model, Arabic pages aren’t read.']);
  eq(L.readersRows(health({ tesseract: 'bad_data' }))[1].segments, ['Installed but can’t be used. Ask your administrator to run ', { code: 'folio doctor' }, '.']);
  eq(L.readersRows(health({ tesseract: 'ok' }))[1].segments, ['Ready.']);
  eq(L.readersRows(withModel({ reachable: true, fit: 'pass' }))[2].segments, ['qwen on gpu-box. Answering. Passed its startup check. Confidence check set up.']);
  eq(L.readersRows(withModel({ reachable: null, fit: 'untested', calibrated: 'uncalibrated' }))[2].segments,
    ['qwen on gpu-box. Not contacted yet. Startup check hasn’t finished yet. Its confidence check isn’t set up for this model yet, so every line it reads is marked to check.']);
  eq(L.readersRows(withModel({ reachable: false, fit: 'fail' }))[2].segments, [
    'qwen on gpu-box. Not answering. Lines it should read are marked “Not read”; read the page again later. '
    + 'Failed its startup check, so every line it reads is marked to check. Ask your administrator to run ',
    { code: 'folio doctor' }, '. Confidence check set up.']);
});

test('pageNotices rows', () => {
  const t = (p, ctx = {}) => L.pageNotices(p, { ...readCtx, ...ctx });
  eq(t(page(1, 'unread'), { queued: true }), [['Waiting to be read.']]);
  eq(t(page(1, 'unread')), [['This page isn’t read yet. ', { button: 'readThis', label: 'Read this page' }]]);
  eq(t(page(1, 'unread'), { readingHere: true }), []);
  eq(t(page(1, 'reading')), [['This page is being read in another window. ', { button: 'refresh', label: 'Refresh' }]]);
  eq(t(page(1, 'reading'), { readingHere: true }), []);
  eq(t(page(1, 'read', [line('p1-l1')]), { imageFailed: true }),
    [['The image of this page can’t be shown.'], ['All lines on this page are checked.']]);
  eq(t(page(1, 'failed', [], { error: 'No reader answered' })), [[
    'This page wasn’t read: No reader answered. ', { button: 'readAgain', label: 'Read again' },
    ' Try again later; if it keeps happening, ask your administrator to run ', { code: 'folio doctor' }, '.']]);
  eq(t(page(1, 'failed')), [[
    'This page wasn’t read. ', { button: 'readAgain', label: 'Read again' },
    ' Try again later; if it keeps happening, ask your administrator to run ', { code: 'folio doctor' }, '.']]);
  const pageError = [
    'Part of this page couldn’t be read: Model timed out. ', { button: 'readAgain', label: 'Read again' },
    ' If it keeps happening, ask your administrator to run ', { code: 'folio doctor' }, '.'];
  const bad = page(1, 'read', [line('p1-l1', 'needs_review', ['not_read'])], { error: 'Model timed out.' });
  eq(t(bad), [pageError]);
  eq(t(bad, { modelConfigured: true }), [
    ['The reading model didn’t answer, so some lines weren’t read. They’re marked “Not read”. Try “Read again” later.'],
    pageError]);
  eq(t(page(1, 'read', [line('p1-l1', 'needs_review', ['needs_model_server']), line('p1-l2', 'needs_review', ['needs_model_server'])])), [[
    'This page wasn’t read: no reader for Arabic is set up. If the pages have no Arabic, add the file again and choose “English only”. '
    + 'Otherwise ask your administrator to install Tesseract or set up a reading model (', { code: 'FOLIO_MODEL_URL' }, ').']]);
  eq(t(page(1, 'read', [line('p1-l1', 'needs_review', ['needs_model_server']), line('p1-l2')])), []);
});

test('pageNotices flags, regions and done', () => {
  const t = (p) => L.pageNotices(p, readCtx);
  const ok = [line('p1-l1', 'needs_review')];
  eq(t(page(1, 'read', ok, { flags: ['orientation_uncertain'] })), [['folio isn’t sure which way up this page is. If the image looks sideways or upside down, check every line against it.']]);
  eq(t(page(1, 'read', ok, { flags: ['text_layer_invalid'], invalid_lines: 4 })), [['The text stored in this PDF was damaged on 4 lines, so folio read this page from its image instead.']]);
  eq(t(page(1, 'read', ok, { flags: ['text_layer_invalid'], invalid_lines: 1 })), [['The text stored in this PDF was damaged on 1 line, so folio read this page from its image instead.']]);
  eq(t(page(1, 'read', ok, { flags: ['invisible_text'] })), [['This PDF came with hidden text added by another program. It’s shown here, and every line is marked to check.']]);
  eq(t(page(1, 'read', ok, { flags: ['empty_with_ink'] })), [['This PDF page had no stored text, so folio read it from its image.']]);
  eq(t(page(1, 'read', ok, { flags: ['deadline'] })), [['Reading this page took too long, so the remaining lines weren’t read. Read the page again to try them.']]);
  eq(t(page(1, 'read', ok, { flags: ['mystery'] })), []);
  eq(t(page(1, 'read', ok, { unread_regions: [[0, 0, 1, 1], [2, 2, 3, 3]] })), [['Some parts of this page weren’t read (2). They’re hatched on the image, and their text won’t be in the export.']]);
  eq(t(page(1, 'read', [line('p1-l1', 'confirmed')])), [['All lines on this page are checked.']]);
  eq(t(page(1, 'read', [])), []);
  eq(t(page(1, 'read', ok, { flags: ['deadline', 'orientation_uncertain'], unread_regions: [[0, 0, 1, 1]] })).length, 3);
});

test('stemOf', () => {
  assert.equal(L.stemOf('a.b.pdf'), 'a.b');
  assert.equal(L.stemOf('noext'), 'noext');
  assert.equal(L.stemOf('.pdf'), '.pdf');
});

test('heldNote and evicted', () => {
  const item = (id, filename, has_edits) => ({ document_id: id, filename, has_edits });
  assert.equal(L.heldNote([item('a', 'a.pdf', false), item('b', 'b.pdf', false)]), null);
  const three = (edits) => [item('a', 'old.pdf', edits), item('b', 'b.pdf', true), item('c', 'c.pdf', true)];
  eq(L.heldNote(three(false)), [`folio holds 3 documents. Adding another removes the oldest, “old.pdf”, where no line was checked or changed.`]);
  eq(L.heldNote(three(true)), ['folio holds 3 documents, and lines in the oldest, “old.pdf”, were checked or changed, so a new one won’t fit. Delete a document first; export it before if you need it.']);
  eq(L.evicted(three(false), three(false).slice(1)), [item('a', 'old.pdf', false)]);
  eq(L.evicted(three(false), three(false)), []);
});

test('replacePage, replaceLine, findLine', () => {
  const d = doc([page(1, 'read', [line('p1-l1'), line('p1-l2')]), page(2, 'unread')]);
  const np = page(2, 'read', [line('p2-l1')]);
  const d2 = L.replacePage(d, np);
  assert.equal(d2.pages[1], np);
  assert.notEqual(d2, d);
  assert.equal(d.pages[1].status, 'unread');
  const edited = line('p1-l2', 'edited', [], { final_text: 'new' });
  const d3 = L.replaceLine(d, edited);
  assert.equal(d3.pages[0].lines[1], edited);
  assert.equal(d.pages[0].lines[1].status, 'auto_accepted');
  assert.equal(L.replaceLine(d, line('p9-l1')), d);
  eq(L.findLine(d, 'p1-l2'), { page: d.pages[0], line: d.pages[0].lines[1] });
  assert.equal(L.findLine(d, 'p1-l9'), null);
});

test('mergeReadPage', () => {
  const read = page(1, 'read', [line('p1-l1', 'needs_review'), line('p1-l2', 'needs_review'), line('p1-l3')]);
  const newer = line('p1-l1', 'edited', [], { final_text: 'mine' });
  const older = line('p1-l2', 'edited', [], { final_text: 'old' });
  const merged = L.mergeReadPage(read, { 'p1-l1': { seq: 5, line: newer }, 'p1-l2': { seq: 3, line: older } }, 3);
  assert.equal(merged.lines[0], newer);
  assert.equal(merged.lines[1], read.lines[1]);
  assert.equal(merged.lines[2], read.lines[2]);
  assert.notEqual(merged, read);
  assert.equal(read.lines[0].status, 'needs_review');
});

test('text to save', () => {
  assert.equal(L.normaliseNewlines('a\r\nb\rc\nd'), 'a\nb\nc\nd');
  assert.equal(L.textToSave('a\nb', 'a\r\nb'), 'a\r\nb');
  assert.equal(L.textToSave('a\nb!', 'a\r\nb'), 'a\nb!');
  assert.equal(L.textToSave('same', 'same'), 'same');
});

test('hash round trip', () => {
  const odd = { docId: 'a b&c=d/é+', page: 3, lineId: 'p3-l2' };
  const hash = L.formatHash(odd);
  assert.equal(hash, '#doc=a%20b%26c%3Dd%2F%C3%A9%2B&page=3&line=p3-l2');
  eq(L.parseHash(hash), { doc: odd.docId, page: '3', line: 'p3-l2' });
  eq(L.parseHash('#doc=abc'), { doc: 'abc', page: null, line: null });
  assert.equal(L.parseHash(''), null);
  assert.equal(L.parseHash('#page=2'), null);
  eq(L.parseHash(L.formatHash({ docId: 'abc' })), { doc: 'abc', page: null, line: null });
});

test('partlyReadPages counts pages with unread areas', () => {
  const pg = (regions) => ({ status: 'read', lines: [], ...(regions === undefined ? {} : { unread_regions: regions }) });
  assert.equal(L.partlyReadPages({ pages: [] }), 0);
  assert.equal(L.partlyReadPages({ pages: [pg(undefined), pg([]), pg([[0, 0, 1, 1]]), pg([[0, 0, 1, 1], [2, 2, 3, 3]])] }), 2);
});
