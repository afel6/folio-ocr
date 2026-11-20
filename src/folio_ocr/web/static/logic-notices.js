// Notices about a page, the Readers dialog rows and the export summary. Pure copy builders.
import { fmtNum, msg, plural, withDot, countLines } from './copy.js';
import { docToCheck, partlyReadPages } from './logic-base.js';

const FOLIO_DOCTOR = { code: 'folio doctor' };

/** Join strings and segment arrays with a separator, merging neighbouring strings. */
function concat(parts, sep = ' ') {
  const out = [];
  const push = (seg) => {
    if (typeof seg === 'string' && typeof out[out.length - 1] === 'string') out[out.length - 1] += seg;
    else out.push(seg);
  };
  parts.forEach((part, i) => {
    if (i > 0) push(sep);
    (Array.isArray(part) ? part : [part]).forEach(push);
  });
  return out;
}

const FIXED = {
  queued: ['Waiting to be read.'],
  imageFailed: ['The image of this page can’t be shown.'],
  unread: ['This page isn’t read yet. ', { button: 'readThis', label: 'Read this page' }],
  elsewhere: ['This page is being read in another window. ', { button: 'refresh', label: 'Refresh' }],
  modelDown: ['The reading model didn’t answer, so some lines weren’t read. They’re marked “Not read”. Try “Read again” later.'],
  noReader: [
    'This page wasn’t read: no reader for Arabic is set up. If the pages have no Arabic, add the file again and choose “English only”. '
    + 'Otherwise ask your administrator to install Tesseract or set up a reading model (', { code: 'FOLIO_MODEL_URL' }, ').'],
  done: ['All lines on this page are checked.'],
};

/** The "all lines checked" notice, which is not a note about the page. */
export const PAGE_DONE = FIXED.done;

const PAGE_FLAGS = [
  ['orientation_uncertain', () => ['folio isn’t sure which way up this page is. If the image looks sideways or upside down, check every line against it.']],
  ['text_layer_invalid', (p) => [`The text stored in this PDF was damaged on ${plural(p.invalid_lines ?? 0, 'line')}, so folio read this page from its image instead.`]],
  ['invisible_text', () => ['This PDF came with hidden text added by another program. It’s shown here, and every line is marked to check.']],
  ['empty_with_ink', () => ['This PDF page had no stored text, so folio read it from its image.']],
  ['deadline', () => ['Reading this page took too long, so the remaining lines weren’t read. Read the page again to try them.']],
];

const readAgain = { button: 'readAgain', label: 'Read again' };

function failedNotice(page) {
  const lead = page.error ? `This page wasn’t read: ${withDot(page.error)} ` : 'This page wasn’t read. ';
  return [lead, readAgain, ' Try again later; if it keeps happening, ask your administrator to run ', FOLIO_DOCTOR, '.'];
}

function pageErrorNotice(page) {
  return [`Part of this page couldn’t be read: ${withDot(page.error)} `, readAgain,
    ' If it keeps happening, ask your administrator to run ', FOLIO_DOCTOR, '.'];
}

function statusNotices(page, ctx) {
  const out = [];
  if (ctx.queued) out.push(FIXED.queued);
  if (page.status === 'unread' && !ctx.queued && !ctx.readingHere) out.push(FIXED.unread);
  if (page.status === 'reading' && !ctx.readingHere) out.push(FIXED.elsewhere);
  if (page.status === 'failed') out.push(failedNotice(page));
  if (page.status === 'read' && page.error) {
    if (ctx.modelConfigured && page.lines.some((l) => l.flags.includes('not_read'))) out.push(FIXED.modelDown);
    out.push(pageErrorNotice(page));
  }
  return out;
}

/** Notices for the top of the review panel, in reading order; each is a segment list. */
export function pageNotices(page, ctx) {
  const out = ctx.imageFailed ? [FIXED.imageFailed] : [];
  out.push(...statusNotices(page, ctx));
  const lines = page.lines ?? [];
  if (lines.length > 0 && lines.every((l) => l.flags.includes('needs_model_server'))) out.push(FIXED.noReader);
  for (const [code, build] of PAGE_FLAGS) {
    if (page.flags?.includes(code)) out.push(build(page));
  }
  const regions = page.unread_regions?.length ?? 0;
  if (regions > 0) {
    out.push([`Some parts of this page weren’t read (${fmtNum(regions)}). They’re hatched on the image, and their text won’t be in the export.`]);
  }
  if (page.status === 'read' && lines.length > 0 && lines.every((l) => l.status !== 'needs_review')) out.push(FIXED.done);
  return out;
}

const BAD = new Set(['missing', 'bad_data']);

/** 'ready', 'check' (something needs attention) or 'unknown' (health not available). */
export function readersState(health, healthError = null) {
  if (!health || healthError) return 'unknown';
  const { weights, tesseract, model } = health;
  const tesseractBad = tesseract === 'bad_data' || (tesseract === 'missing' && !model.configured);
  const modelBad = model.reachable === false || model.fit === 'fail';
  return BAD.has(weights) || tesseractBad || modelBad ? 'check' : 'ready';
}

const UNCAL = 'Ready. Its confidence check isn’t set up yet, so every line it reads is marked to check.';

function scanRow(status) {
  const rows = {
    ok: ['Ready.'],
    uncalibrated: [UNCAL],
    missing: ['Its files are missing. Ask your administrator to run ', { code: 'folio setup' }, '.'],
    bad_data: ['Its files can’t be used. Ask your administrator to run ', FOLIO_DOCTOR, '.'],
  };
  return rows[status] ?? [];
}

function tesseractRow(status) {
  const rows = {
    missing: ['Not installed (optional). Without it or a reading model, Arabic pages aren’t read.'],
    bad_data: ['Installed but can’t be used. Ask your administrator to run ', FOLIO_DOCTOR, '.'],
  };
  return rows[status] ?? scanRow(status);
}

const REACHABLE = new Map([
  [null, 'Not contacted yet.'], [true, 'Answering.'],
  [false, 'Not answering. Lines it should read are marked “Not read”; read the page again later.'],
]);
const FIT = {
  untested: 'Startup check hasn’t finished yet.',
  pass: 'Passed its startup check.',
  fail: ['Failed its startup check, so every line it reads is marked to check. Ask your administrator to run ', FOLIO_DOCTOR, '.'],
};
const CALIBRATED = {
  ok: 'Confidence check set up.',
  uncalibrated: 'Its confidence check isn’t set up for this model yet, so every line it reads is marked to check.',
};

function modelRow(model) {
  if (!model.configured) return ['Not set up (optional).'];
  const sentences = [REACHABLE.get(model.reachable ?? null), FIT[model.fit], CALIBRATED[model.calibrated]];
  return concat([`${model.name} on ${model.host}.`, ...sentences.filter((part) => part !== undefined)]);
}

/** Rows for the Readers dialog; empty when health is not available. */
export function readersRows(health) {
  if (!health) return [];
  return [
    { name: 'Scan reader', segments: scanRow(health.weights) },
    { name: 'Tesseract (Arabic and English)', segments: tesseractRow(health.tesseract) },
    { name: 'Reading model', segments: modelRow(health.model) },
  ];
}

function listPageNumbers(nums) {
  const shown = nums.map(fmtNum);
  if (shown.length === 1) return `Page ${shown[0]} is`;
  return `Pages ${shown.slice(0, -1).join(', ')} and ${shown.at(-1)} are`;
}

/** The sentence block above the export buttons. */
export function exportSummary(doc, readingPages) {
  const reading = [...readingPages].sort((a, b) => a - b);
  const toCheckCount = docToCheck(doc);
  const parts = [toCheckCount === 0 ? countLines(0)
    : `${countLines(toCheckCount)} ${toCheckCount === 1
      ? 'It’s exported as read, and the Word file highlights it.'
      : 'They’re exported as read, and the Word file highlights them.'}`];
  const empty = doc.pages.filter((p) => (p.status === 'unread' || p.status === 'failed') && !reading.includes(p.index)).length;
  if (empty > 0) {
    parts.push(`${plural(empty, 'page')} ${empty === 1 ? 'isn’t read, so it’ll' : 'aren’t read, so they’ll'} be empty in the export.`);
  }
  const partial = partlyReadPages(doc);
  if (partial > 0) parts.push(msg.partlyRead(partial));
  if (reading.length > 0) parts.push(`${listPageNumbers(reading)} still being read and won’t be in an export made now.`);
  return parts.join(' ');
}
