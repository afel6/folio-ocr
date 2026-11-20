import test from 'node:test';
import assert from 'node:assert/strict';
import {
  fmtNum, plural, withDot, quoted, elapsed, LANGUAGE_NAMES, statusLabel, flagCopy, msg, errorMessage,
} from '../../src/folio_ocr/web/static/copy.js';
import { ApiError } from '../../src/folio_ocr/web/static/errors.js';

const eq = assert.deepEqual;
const seg = (v) => (Array.isArray(v) ? v : [v]);
const http = (status, line = null) => new ApiError('http', status, line);

test('number helpers', () => {
  assert.equal(fmtNum(1234), '1,234');
  assert.equal(plural(1, 'page'), '1 page');
  assert.equal(plural(3, 'page'), '3 pages');
  assert.equal(plural(1200, 'line'), '1,200 lines');
  assert.equal(withDot('Bad file'), 'Bad file.');
  assert.equal(withDot('Bad file.'), 'Bad file.');
  assert.equal(quoted('a'), '“a”');
  assert.equal(elapsed(12000), '0:12');
  assert.equal(elapsed(65000), '1:05');
  assert.equal(elapsed(720000), '12:00');
});

test('language names and status labels', () => {
  assert.equal(LANGUAGE_NAMES['en+ar'], 'English and Arabic');
  eq(statusLabel('needs_review'), { label: 'To check', glyph: '!' });
  eq(statusLabel('auto_accepted'), { label: 'Accepted', glyph: '' });
  eq(statusLabel('edited'), { label: 'Edited', glyph: '✎' });
  eq(statusLabel('confirmed'), { label: 'Confirmed', glyph: '✓' });
  eq(statusLabel('weird'), { label: 'weird', glyph: '' });
});

test('flagCopy covers every line code and unknowns', () => {
  const rows = {
    low_confidence: 'Unsure reading', mixed_order: 'Arabic and English mixed', visual_order: 'Word order',
    text_layer_invalid: 'Damaged PDF text', needs_model_server: 'Not read: no reader',
    model_self_test_failed: 'Model check failed', truncated: 'May be cut short', no_transcription: 'No text returned',
    format_noise: 'Extra text', not_read: 'Not read', script_mismatch: 'Unexpected letters',
    orientation_uncertain: 'Page may be turned', handwritten: 'Handwriting', empty: 'No text found',
    repeated: 'Repeated text', presentation_forms: 'Display-form letters', lookalike_chars: 'Look-alike letters',
    invalid_text: 'Broken characters', cid_codes: 'Unreadable PDF codes', private_use: 'Private symbols',
  };
  for (const [code, label] of Object.entries(rows)) assert.equal(flagCopy(code).label, label, code);
  assert.equal(flagCopy('low_confidence').sentence, 'The reader wasn’t sure of this text. Compare it with the image.');
  assert.equal(flagCopy('not_read').sentence, 'This line couldn’t be read this time. Read the page again later, or type the text from the image.');
  assert.equal(flagCopy('orientation_uncertain').sentence, 'folio isn’t sure which way up this page is, so this text may be wrong. Check it against the image.');
  eq(flagCopy('zzz'), { label: 'Check this line', sentence: 'folio marked this line to check. Compare it with the image.' });
});

test('plain messages', () => {
  assert.equal(msg.uploading('a.pdf'), 'Adding a.pdf…');
  assert.equal(msg.added('a.pdf', 5), 'a.pdf added: 5 pages.');
  assert.equal(msg.added('a.pdf', 1), 'a.pdf added: 1 page.');
  assert.equal(msg.heldNoEdits('old.pdf'), 'folio holds 3 documents. Adding another removes the oldest, “old.pdf”, where no line was checked or changed.');
  assert.equal(msg.heldWithEdits('old.pdf'), 'folio holds 3 documents, and lines in the oldest, “old.pdf”, were checked or changed, so a new one won’t fit. Delete a document first; export it before if you need it.');
  assert.equal(msg.storeFull('a.pdf'), 'Can’t add a.pdf: folio is full. Delete a document first to make room; export it before if you need it.');
  assert.equal(msg.evicted('old.pdf'), '“old.pdf” was removed to make room. None of its lines had been checked or changed.');
  assert.equal(msg.badFile('a.pdf', 'File is too big'), 'Can’t add a.pdf: File is too big.');
  assert.equal(msg.readingStatus(2, 5), 'Reading page 2 of 5…');
  assert.equal(msg.readingBar(2), 'Reading page 2…');
  assert.equal(msg.progressLabel(2), 'Reading page 2');
  assert.equal(msg.readDone(2, 14, 3, false), 'Page 2 read: 14 lines, 3 to check.');
  assert.equal(msg.readDone(2, 1, 1, true), 'Page 2 read: 1 line, 1 to check. See the notes about this page.');
  assert.equal(msg.modelNoAnswer(2), 'The reading model didn’t answer, so page 2 wasn’t read. Try again later, or ask your administrator to check the reading model.');
  assert.equal(msg.cantRead(2, 'No picture'), 'Can’t read page 2: No picture.');
  assert.equal(msg.weights(2, 'Files missing; run folio setup.'), 'Can’t read page 2: Files missing; run folio setup. Ask your administrator to fix this.');
  assert.equal(msg.refused(), 'folio refused this request because it came from another page. Open folio from its own address and try again.');
  assert.equal(msg.saved(5), 'Line 5 saved.');
  assert.equal(msg.confirmed(5), 'Line 5 confirmed.');
  assert.equal(msg.saveFailed(5, null), 'Couldn’t save line 5: folio didn’t answer. Your text is still in the box.');
  assert.equal(msg.saveFailed(5, 'Text is bad'), 'Couldn’t save line 5: Text is bad. Your text is still in the box.');
  assert.equal(msg.nothingLeft(0), 'No lines left to check on read pages.');
  assert.equal(msg.nothingLeft(2), 'No lines left to check on read pages. 2 pages aren’t read yet.');
  assert.equal(msg.nothingLeft(1), 'No lines left to check on read pages. 1 page isn’t read yet.');
  assert.equal(msg.docGone('a.pdf'), '“a.pdf” is no longer in folio. It was deleted, removed to make room, or folio restarted.');
  assert.equal(msg.exported('a', 'docx', 3), 'Exported a.docx. 3 lines still need checking.');
  assert.equal(msg.exported('a', 'txt', 1), 'Exported a.txt. 1 line still needs checking.');
  assert.equal(msg.exported('a', 'txt', 0), 'Exported a.txt. No lines left to check.');
  assert.equal(msg.exported('a', 'txt', 0, 0), 'Exported a.txt. No lines left to check.');
  assert.equal(msg.exported('a', 'json', 2, 1),
    'Exported a.json. 2 lines still need checking. Parts of 1 page weren’t read, so their text isn’t in the export.');
  assert.equal(msg.exported('a', 'docx', 0, 3),
    'Exported a.docx. No lines left to check. Parts of 3 pages weren’t read, so their text isn’t in the export.');
  assert.equal(msg.partlyRead(1), 'Parts of 1 page weren’t read, so their text isn’t in the export.');
  assert.equal(msg.exportFailed('a.pdf', null), 'Couldn’t export a.pdf: folio didn’t answer. Try again.');
  assert.equal(msg.exportFailed('a.pdf', 'Boom'), 'Couldn’t export a.pdf: Boom. Try again.');
  assert.equal(msg.deleted('a.pdf'), 'a.pdf deleted.');
  assert.equal(msg.deleteTitle('a.pdf'), 'Delete a.pdf?');
  assert.equal(msg.deleteBody(true), 'folio will forget this document and the lines you checked or changed. Export first if you need them. This can’t be undone.');
  assert.equal(msg.deleteBody(false), 'folio will forget this document. This can’t be undone.');
  assert.equal(msg.unsavedDraft(5), 'Save (Enter) or cancel (Esc) your change to line 5 first.');
  assert.equal(msg.modelBanner('gpu-box'), 'Pages you read are sent to a reading model on another computer (gpu-box). Don’t add documents that must stay on this machine.');
  assert.equal(msg.pageTitle(2, 5, 'a.pdf'), 'Page 2 of 5 – a.pdf – folio');
  assert.equal(msg.pageOption(2, 5, 3), 'Page 2 of 5 – 3 to check');
  assert.equal(msg.pageOption(2, 5, 0), 'Page 2 of 5');
});

test('docMeta', () => {
  const iso = '2026-11-19T10:42:00';
  const time = new Intl.DateTimeFormat('en', { timeStyle: 'short' }).format(new Date(iso));
  assert.equal(msg.docMeta('en+ar', 5, iso), `English and Arabic, 5 pages, added ${time}`);
  assert.equal(msg.docMeta('en', 1, 'nonsense'), 'English, 1 page');
  assert.equal(msg.docMeta('ar', 2, null), 'Arabic, 2 pages');
});

test('segment messages', () => {
  eq(msg.readBusy(), ['Another page is being read right now, maybe in another window. Try again when it finishes. ', { button: 'tryAgain', label: 'Try again' }]);
  eq(msg.unreachable(), ['Can’t reach folio. Check that it’s still running (', { code: 'folio serve' }, '), then reload this page.']);
  eq(msg.generic(500), ['Something went wrong in folio (error 500). Try again. If it keeps happening, ask your administrator to run ', { code: 'folio doctor' }, '.']);
  eq(msg.generic(null), ['Something went wrong in folio. Try again. If it keeps happening, ask your administrator to run ', { code: 'folio doctor' }, '.']);
});

test('errorMessage rules', () => {
  const net = new ApiError('network');
  eq(errorMessage(new Error('x'), {}), seg(msg.generic(null)));
  eq(errorMessage(net, { action: 'save', line: 5 }), seg(msg.saveFailed(5, null)));
  eq(errorMessage(net, { action: 'export', file: 'a.pdf' }), seg(msg.exportFailed('a.pdf', null)));
  eq(errorMessage(net, { action: 'read' }), seg(msg.unreachable()));
  eq(errorMessage(http(403), { action: 'save' }), seg(msg.refused()));
  eq(errorMessage(http(404), { action: 'read', name: 'a.pdf' }), seg(msg.docGone('a.pdf')));
  eq(errorMessage(http(409), { action: 'upload', file: 'a.pdf' }), seg(msg.storeFull('a.pdf')));
  eq(errorMessage(http(400, 'Too big'), { action: 'upload', file: 'a.pdf' }), seg(msg.badFile('a.pdf', 'Too big')));
  eq(errorMessage(http(400), { action: 'upload', file: 'a.pdf' }), seg(msg.generic(400)));
  eq(errorMessage(http(409), { action: 'read', page: 2 }), seg(msg.readBusy()));
  eq(errorMessage(http(502), { action: 'read', page: 2 }), seg(msg.modelNoAnswer(2)));
  eq(errorMessage(http(400, 'No'), { action: 'read', page: 2 }), seg(msg.cantRead(2, 'No')));
  eq(errorMessage(http(503, 'Run folio setup'), { action: 'read', page: 2 }), seg(msg.weights(2, 'Run folio setup')));
  eq(errorMessage(http(400), { action: 'read', page: 2 }), seg(msg.generic(400)));
  eq(errorMessage(http(503), { action: 'read', page: 2 }), seg(msg.generic(503)));
  eq(errorMessage(new ApiError('badBody', 200), { action: 'save', line: 5 }), seg(msg.saveFailed(5, null)));
  eq(errorMessage(http(400, 'Bad text'), { action: 'save', line: 5 }), seg(msg.saveFailed(5, 'Bad text')));
  eq(errorMessage(http(500, 'Oops'), { action: 'export', file: 'a.pdf' }), seg(msg.exportFailed('a.pdf', 'Oops')));
  eq(errorMessage(http(405), { action: 'list' }), seg(msg.generic(405)));
});
