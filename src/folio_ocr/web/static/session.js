// Shared pieces of the running page: the fetch used for every call, reply ordering, saved
// lines, forgotten documents, selection helpers and the URL hash.
import { $ } from './dom.js';
import { errorMessage, msg } from './copy.js';
import { findLine, formatHash, lineNumber, normaliseNewlines } from './logic.js';
import { getState, setState } from './store.js';

export const fetchFn = (url, init) => window.fetch(url, init);

let seq = 0;
/** A number that grows with every read sent and every line saved, to order their replies. */
export const stamp = () => {
  seq += 1;
  return seq;
};

let saved = {};
/** Lines saved in this tab, by document, with the stamp of their save. */
export const savedFor = (docId) => saved[docId] ?? {};
export function recordSaved(docId, line) {
  saved = { ...saved, [docId]: { ...savedFor(docId), [line.id]: { seq: stamp(), line } } };
}

let removed = new Set();
export const isAlive = (docId) => !removed.has(docId);
export function markRemoved(docId) {
  removed = new Set([...removed, docId]);
  const { [docId]: _gone, ...rest } = saved;
  saved = rest;
}

let pending = new Set();
/** Run a save and remember it until it settles, so an export can wait for it. */
export async function trackSave(promise) {
  pending = new Set([...pending, promise]);
  try {
    return await promise;
  } finally {
    pending = new Set([...pending].filter((p) => p !== promise));
  }
}
export const pendingSaves = () => [...pending];

export const currentDoc = (state = getState()) => state.docs[state.docId] ?? null;
export function currentPage(state = getState()) {
  const doc = currentDoc(state);
  return doc ? doc.pages[state.page - 1] ?? null : null;
}
export function currentLine(state = getState()) {
  const page = currentPage(state);
  return page ? page.lines.find((l) => l.id === state.lineId) ?? null : null;
}

/** The number shown for a line ("Line 5"). */
export function lineK(page, line) {
  return lineNumber(line, page.lines.findIndex((l) => l.id === line.id));
}

/** The line to select on a page: its first line to check, else its first line. */
export function defaultLineId(page) {
  if (!page || page.lines.length === 0) return null;
  return (page.lines.find((l) => l.status === 'needs_review') ?? page.lines[0]).id;
}

/** True when the open edit box holds text that differs from the stored text. */
export function draftChanged(state = getState()) {
  const { draft } = state;
  if (!draft) return false;
  const doc = state.docs[draft.docId];
  const line = doc ? findLine(doc, draft.lineId)?.line : null;
  return Boolean(line) && normaliseNewlines(line.final_text) !== draft.value;
}

/** Stop a move while the edit box holds unsaved text: show why and put focus in the box. */
export function blockedByDraft() {
  const state = getState();
  if (!draftChanged(state)) return false;
  const page = currentPage(state);
  const line = page?.lines.find((l) => l.id === state.draft.lineId);
  const k = line ? lineK(page, line) : '';
  setState({ alert: [msg.unsavedDraft(k)] });
  focus('editor');
  return true;
}

export function showError(err, ctx) {
  setState({ alert: errorMessage(err, ctx) });
}

export function focus(id) {
  const el = $(id);
  if (el && !el.hidden) el.focus();
}

export function writeHash(state = getState()) {
  const hash = state.screen === 'doc' && state.docId
    ? formatHash({ docId: state.docId, page: state.page, lineId: state.lineId })
    : '';
  if (hash === window.location.hash) return;
  window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}${hash}`);
}
