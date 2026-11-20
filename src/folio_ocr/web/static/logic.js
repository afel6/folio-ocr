// Review-screen logic: finding lines, merging reads, saving text and the URL hash; re-exports logic-base, -notices and -ui.
import { msg } from './copy.js';

export * from './logic-base.js';
export * from './logic-notices.js';
export * from './logic-ui.js';

const allLines = (doc) => doc.pages.flatMap((page, p) => page.lines.map((l) => ({ page: p, lineId: l.id, status: l.status })));

/** The next (dir +1) or previous (dir -1) line to check, wrapping round; the current line only if it is the last one. */
export function nextToCheck(doc, pageIndex, lineId, dir) {
  const flat = allLines(doc);
  const n = flat.length;
  if (n === 0) return null;
  const at = flat.findIndex((e) => e.page === pageIndex && e.lineId === lineId);
  const start = at >= 0 ? at : (dir > 0 ? -1 : n);
  for (let step = 1; step <= n; step += 1) {
    const hit = flat[(((start + dir * step) % n) + n) % n];
    if (hit.status === 'needs_review') return { page: hit.page, lineId: hit.lineId };
  }
  return null;
}

export function stemOf(filename) {
  const dot = filename.lastIndexOf('.');
  return dot > 0 ? filename.slice(0, dot) : filename;
}

/** The note about holding 3 documents, or null while there is room. */
export function heldNote(list) {
  if (list.length < 3) return null;
  const oldest = list[0];
  const name = oldest.filename;
  return [oldest.has_edits ? msg.heldWithEdits(name) : msg.heldNoEdits(name)];
}

/** Items of `before` that are missing from `after`. */
export function evicted(before, after) {
  const kept = new Set(after.map((item) => item.document_id));
  return before.filter((item) => !kept.has(item.document_id));
}

export const replacePage = (doc, page) => ({
  ...doc,
  pages: doc.pages.map((p) => (p.index === page.index ? page : p)),
});

export function replaceLine(doc, line) {
  const n = Number(/^p(\d+)-l/.exec(line.id)?.[1]);
  const page = doc.pages.find((p) => p.index === n);
  if (!page) return doc;
  return replacePage(doc, { ...page, lines: page.lines.map((l) => (l.id === line.id ? line : l)) });
}

export function findLine(doc, lineId) {
  for (const page of doc.pages) {
    const line = page.lines.find((l) => l.id === lineId);
    if (line) return { page, line };
  }
  return null;
}

/** A freshly read page, except lines saved after the read was sent keep their saved version. */
export function mergeReadPage(page, saved, readSeq) {
  const lines = page.lines.map((l) => (saved[l.id] && saved[l.id].seq > readSeq ? saved[l.id].line : l));
  return { ...page, lines };
}

export const normaliseNewlines = (s) => s.replace(/\r\n?/g, '\n');

/** Send the stored text untouched when the box only differs by line-end style. */
export const textToSave = (boxValue, stored) => (normaliseNewlines(stored) === boxValue ? stored : boxValue);

export function parseHash(hash) {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const doc = params.get('doc');
  return doc ? { doc, page: params.get('page'), line: params.get('line') } : null;
}

export function formatHash({ docId, page, lineId }) {
  const parts = [['doc', docId], ['page', page], ['line', lineId]]
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${k}=${encodeURIComponent(v)}`);
  return `#${parts.join('&')}`;
}
