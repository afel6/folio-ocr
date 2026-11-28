// Every word the review screen shows, as pure functions. No DOM, no network.
import { ApiError, isReadersBusy } from './errors.js';

const NUM = new Intl.NumberFormat('en');
const TIME = new Intl.DateTimeFormat('en', { timeStyle: 'short' });

export const fmtNum = (n) => NUM.format(n);
export const plural = (n, one, other = `${one}s`) => `${fmtNum(n)} ${n === 1 ? one : other}`;
export const withDot = (s) => (s.endsWith('.') ? s : `${s}.`);
export const quoted = (name) => `“${name}”`;

export function elapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

export const LANGUAGE_NAMES = { 'en+ar': 'English and Arabic', en: 'English', ar: 'Arabic' };

const STATUS = {
  needs_review: { label: 'To check', glyph: '!' },
  auto_accepted: { label: 'Accepted', glyph: '' },
  edited: { label: 'Edited', glyph: '✎' },
  confirmed: { label: 'Confirmed', glyph: '✓' },
};

export function statusLabel(status) {
  return { ...(STATUS[status] ?? { label: status, glyph: '' }) };
}

const FLAGS = {
  low_confidence: ['Unsure reading', 'The reader wasn’t sure of this text. Compare it with the image.'],
  mixed_order: ['Arabic and English mixed', 'This line mixes Arabic and English. Check that the words are in the right order.'],
  visual_order: ['Word order', 'The PDF may store these words in a different order from the page. Check the order against the image.'],
  text_layer_invalid: ['Damaged PDF text', 'Some characters stored in the PDF for this line aren’t real letters. Check the text against the image.'],
  needs_model_server: ['Not read: no reader', 'No reader for this language is set up, so this line wasn’t read. Type the text from the image, or see Readers.'],
  model_self_test_failed: ['Model check failed', 'The reading model failed its startup check, so this text may be wrong. Check it closely.'],
  truncated: ['May be cut short', 'The reading stopped early, so the end of this line may be missing.'],
  no_transcription: ['No text returned', 'The reading model returned no text for this line. Type the text from the image if there is any.'],
  format_noise: ['Extra text', 'The reading model may have added notes or symbols that aren’t on the page. Remove anything the image doesn’t show.'],
  not_read: ['Not read', 'This line couldn’t be read this time. Read the page again later, or type the text from the image.'],
  script_mismatch: ['Unexpected letters', 'This line has letters that don’t match the languages chosen for this document. Check them against the image.'],
  orientation_uncertain: ['Page may be turned', 'folio isn’t sure which way up this page is, so this text may be wrong. Check it against the image.'],
  handwritten: ['Handwriting', 'This line looks handwritten. Check names, dates and numbers closely.'],
  empty: ['No text found', 'No text was found in this line’s box. Type it if the image shows any.'],
  repeated: ['Repeated text', 'Part of this line repeats several times. Check that it isn’t doubled by mistake.'],
  presentation_forms: ['Display-form letters', 'Some Arabic letters are stored in a display-only form. They look right but may not search or sort correctly; retype them if that matters.'],
  lookalike_chars: ['Look-alike letters', 'Some letters may come from another alphabet that looks the same, such as Greek, Cyrillic or Persian forms. Check the spelling.'],
  invalid_text: ['Broken characters', 'Some characters couldn’t be stored and show as �. Retype them from the image.'],
  cid_codes: ['Unreadable PDF codes', 'The PDF stores some characters here as codes with no letter. Check the text against the image.'],
  private_use: ['Private symbols', 'The PDF uses private symbols here that aren’t real letters. Check the text against the image.'],
};
const UNKNOWN_FLAG = ['Check this line', 'folio marked this line to check. Compare it with the image.'];

export function flagCopy(code) {
  const [label, sentence] = Object.hasOwn(FLAGS, code) ? FLAGS[code] : UNKNOWN_FLAG;
  return { label, sentence };
}

const noLines = 'No lines left to check.';
const answerLine = (line) => (line ? withDot(line) : 'folio didn’t answer.');
const countLines = (n) => (n === 0 ? noLines : `${fmtNum(n)} ${n === 1 ? 'line still needs' : 'lines still need'} checking.`);

function docMeta(languages, pageCount, createdAtIso) {
  const base = `${LANGUAGE_NAMES[languages] ?? languages}, ${plural(pageCount, 'page')}`;
  const date = new Date(createdAtIso);
  return createdAtIso && !Number.isNaN(date.getTime()) ? `${base}, added ${TIME.format(date)}` : base;
}

export const msg = {
  uploading: (file) => `Adding ${file}…`,
  added: (file, n) => `${file} added: ${plural(n, 'page')}.`,
  heldNoEdits: (name) => `folio holds 3 documents. Adding another removes the oldest, ${quoted(name)}, where no line was checked or changed.`,
  heldWithEdits: (name) => `folio holds 3 documents, and lines in the oldest, ${quoted(name)}, were checked or changed, so a new one won’t fit. Delete a document first; export it before if you need it.`,
  storeFull: (file) => `Can’t add ${file}: folio is full. Delete a document first to make room; export it before if you need it.`,
  evicted: (name) => `${quoted(name)} was removed to make room. None of its lines had been checked or changed.`,
  badFile: (file, line) => `Can’t add ${file}: ${withDot(line)}`,
  addBusy: (file) => `Can’t add ${file}: folio is busy reading other files. Try again in a moment.`,
  addTooSlow: (file) => `Can’t add ${file}: the upload took too long. Check your connection and try again.`,
  readingStatus: (n, total) => `Reading page ${fmtNum(n)} of ${fmtNum(total)}…`,
  readingBar: (n) => `Reading page ${fmtNum(n)}…`,
  progressLabel: (n) => `Reading page ${fmtNum(n)}`,
  readDone: (n, lines, toCheck, hasNotes) =>
    `Page ${fmtNum(n)} read: ${plural(lines, 'line')}, ${fmtNum(toCheck)} to check.${hasNotes ? ' See the notes about this page.' : ''}`,
  readBusy: () => ['Another page is being read right now, maybe in another window. Try again when it finishes. ', { button: 'tryAgain', label: 'Try again' }],
  modelNoAnswer: (n) => `The reading model didn’t answer, so page ${fmtNum(n)} wasn’t read. Try again later, or ask your administrator to check the reading model.`,
  readersBusy: (n) => `Can’t read page ${fmtNum(n)}: folio is busy reading other files. Try again in a moment.`,
  cantRead: (n, line) => `Can’t read page ${fmtNum(n)}: ${withDot(line)}`,
  weights: (n, line) => `Can’t read page ${fmtNum(n)}: ${withDot(line)} Ask your administrator to fix this.`,
  refused: () => 'folio refused this request because it came from another page. Open folio from its own address and try again.',
  saved: (n) => `Line ${fmtNum(n)} saved.`,
  confirmed: (n) => `Line ${fmtNum(n)} confirmed.`,
  saveFailed: (n, line) => `Couldn’t save line ${fmtNum(n)}: ${answerLine(line)} Your text is still in the box.`,
  nothingLeft: (unreadPages) => `No lines left to check on read pages.${unreadPages > 0 ? ` ${plural(unreadPages, 'page')} ${unreadPages === 1 ? 'isn’t' : 'aren’t'} read yet.` : ''}`,
  docGone: (name) => `${quoted(name)} is no longer in folio. It was deleted, removed to make room, or folio restarted.`,
  unreachable: () => ['Can’t reach folio. Check that it’s still running (', { code: 'folio serve' }, '), then reload this page.'],
  generic: (code) => [`Something went wrong in folio${code === null ? '' : ` (error ${code})`}. Try again. If it keeps happening, ask your administrator to run `, { code: 'folio doctor' }, '.'],
  exported: (stem, ext, lines, partPages = 0) => [`Exported ${stem}.${ext}.`, countLines(lines), partPages > 0 ? msg.partlyRead(partPages) : '']
    .filter(Boolean).join(' '),
  partlyRead: (pages) => `Parts of ${plural(pages, 'page')} weren’t read, so their text isn’t in the export.`,
  exportFailed: (file, line) => `Couldn’t export ${file}: ${answerLine(line)} Try again.`,
  deleted: (file) => `${file} deleted.`,
  deleteTitle: (file) => `Delete ${file}?`,
  deleteBody: (hasEdits) => (hasEdits
    ? 'folio will forget this document and the lines you checked or changed. Export first if you need them. This can’t be undone.'
    : 'folio will forget this document. This can’t be undone.'),
  unsavedDraft: (n) => `Save (Enter) or cancel (Esc) your change to line ${fmtNum(n)} first.`,
  modelBanner: (host) => `Pages you read are sent to a reading model on another computer (${host}). Don’t add documents that must stay on this machine.`,
  pageTitle: (n, total, file) => `Page ${fmtNum(n)} of ${fmtNum(total)} – ${file} – folio`,
  docMeta,
  pageOption: (n, total, toCheck) => `Page ${fmtNum(n)} of ${fmtNum(total)}${toCheck > 0 ? ` – ${fmtNum(toCheck)} to check` : ''}`,
};

export { countLines };

const asSegments = (value) => (Array.isArray(value) ? value : [value]);

function readError(err, ctx) {
  const { status, serverLine } = err;
  if (status === 409) return msg.readBusy();
  if (status === 502) return msg.modelNoAnswer(ctx.page);
  if (status === 400 && serverLine) return msg.cantRead(ctx.page, serverLine);
  if (isReadersBusy(err)) return msg.readersBusy(ctx.page);
  if (status === 503 && serverLine) return msg.weights(ctx.page, serverLine);
  return null;
}

function uploadError(err, file) {
  if (err.status === 409) return msg.storeFull(file);
  if (err.status === 408) return msg.addTooSlow(file);
  if (isReadersBusy(err)) return msg.addBusy(file);
  return err.status === 400 && err.serverLine ? msg.badFile(file, err.serverLine) : null;
}

function actionError(err, ctx) {
  switch (ctx.action) {
    case 'upload': return uploadError(err, ctx.file);
    case 'read': return readError(err, ctx);
    case 'save': return msg.saveFailed(ctx.line, err.serverLine);
    case 'export': return msg.exportFailed(ctx.file, err.serverLine);
    default: return null;
  }
}

/** Turn any failure into friendly copy (segments); never leaks a stack trace. */
export function errorMessage(err, ctx = {}) {
  if (!(err instanceof ApiError)) return asSegments(msg.generic(null));
  if (err.kind === 'network') {
    if (ctx.action === 'save') return asSegments(msg.saveFailed(ctx.line, null));
    if (ctx.action === 'export') return asSegments(msg.exportFailed(ctx.file, null));
    return asSegments(msg.unreachable());
  }
  if (err.status === 403) return asSegments(msg.refused());
  if (err.status === 404) return asSegments(msg.docGone(ctx.name));
  return asSegments(actionError(err, ctx) ?? msg.generic(err.status));
}
