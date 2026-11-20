// Decisions the screen draws from: list movement, close-up crop, navigation text, labels, key mapping, and drawing every part.
import { fmtNum } from './copy.js';
import { docToCheck, partlyReadPages } from './logic-base.js';

const PAD = 8;

/** The index a list key moves to, or null for a key that does not move. */
export function moveTarget(key, index, count) {
  const step = { ArrowDown: 1, ArrowUp: -1, PageDown: 10, PageUp: -10 }[key];
  if (key === 'Home') return 0;
  if (key === 'End') return count - 1;
  if (step === undefined) return null;
  return Math.min(count - 1, Math.max(0, index + step));
}

/** The action a key on the line list runs, as [name, ...args], or null for a key the list leaves alone. */
export function listKeyAction(event, lines, lineId) {
  if (event.ctrlKey || event.altKey || event.metaKey) return null;
  const { key } = event;
  const index = lines.findIndex((l) => l.id === lineId);
  const target = lines.length > 0 ? moveTarget(key, index, lines.length) : null;
  if (target !== null) return ['selectLine', lines[target].id, { focusList: true }];
  if (key === 'n' || key === 'N') return ['next', event.shiftKey ? -1 : 1];
  if (key === 'c' || key === 'C') return ['confirm'];
  if (key === 'e' || key === 'E' || key === 'Enter') return ['openEditor'];
  return null;
}

/** The action a key in the edit box runs: Enter saves unless Shift is held or text is being composed; Esc cancels. */
export function editorKeyAction(event) {
  if (event.key === 'Enter' && !event.isComposing && !event.shiftKey) return ['save'];
  if (event.key === 'Escape') return ['cancelEdit'];
  return null;
}

/** Draw every part on its own, so one failure leaves the others drawn; true when any part failed. */
export function drawParts(parts, state) {
  let failed = false;
  for (const part of parts) {
    try {
      part.render(state);
    } catch {
      failed = true;
    }
  }
  return failed;
}

/** The error message shows once per run of failed draws: when this draw failed and the one before did not. */
export const alertOnDrawFailure = (failed, failedBefore) => failed && !failedBefore;

/** The line's box with a margin, clamped to the page, as [x, y, width, height]. */
export function closeupBox(page, line) {
  const W = page.width;
  const H = page.height;
  const b = line.box;
  const x0 = Math.max(0, b[0] - PAD);
  const y0 = Math.max(0, b[1] - PAD);
  const x1 = Math.min(W, b[2] + PAD);
  const y1 = Math.min(H, b[3] + PAD);
  if (x1 <= x0 || y1 <= y0) return [0, 0, W, H];
  return [x0, y0, x1 - x0, y1 - y0];
}

export function isRemoteHost(hostname) {
  return hostname !== 'localhost' && !hostname.startsWith('127.') && hostname !== '[::1]';
}

/**
 * The count beside a document in the navigation: "<n> to check", "done" or nothing. Like the
 * list's unresolved_count, it counts lines to check plus pages with unread areas.
 */
export function countText(item, doc) {
  const n = doc ? docToCheck(doc) + partlyReadPages(doc) : item.unresolved_count;
  const pages = doc ? doc.pages : item.pages;
  if (n > 0) return `${fmtNum(n)} to check`;
  return pages.every((p) => p.status === 'read') ? 'done' : '';
}

/** Numbers of the pages being read now: marked reading, or in flight in the open document. */
export function readingPages(state, doc) {
  const flying = state.inFlight && state.inFlight.docId === state.docId ? state.inFlight.page : null;
  return doc.pages.filter((p) => p.status === 'reading' || p.index === flying).map((p) => p.index);
}

/** After a delete: the next document in the list, else the first one left, else null. */
export function nextDocId(list, docId) {
  const at = list.findIndex((item) => item.document_id === docId);
  const rest = list.filter((item) => item.document_id !== docId);
  if (rest.length === 0) return null;
  return (rest[at] ?? rest[0]).document_id;
}

export function clampPage(doc, page) {
  const n = Number.parseInt(page, 10);
  return Number.isInteger(n) && n >= 1 && n <= doc.pages.length ? n : 1;
}

/** The read button's text for a page, or null when no button is offered. */
export function readButtonLabel(page, { queued, readingHere }) {
  if (queued || readingHere) return null;
  if (page.status === 'unread') return 'Read this page';
  return page.status === 'read' || page.status === 'failed' ? 'Read again' : null;
}

/** True when the page waits in the read queue; a paused queue waits on the user, so nothing is queued. */
export function isQueued(state, docId, n) {
  return !state.paused && state.queue.some((q) => q.docId === docId && q.page === n);
}
