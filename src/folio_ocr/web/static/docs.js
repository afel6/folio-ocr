// Documents: the list, opening one, moving between pages, forgetting one that is gone.
import { ApiError, STALE, callJson, paths } from './api.js';
import { errorMessage, msg } from './copy.js';
import { clampPage } from './logic.js';
import { getState, setState } from './store.js';
import {
  blockedByDraft, currentDoc, defaultLineId, fetchFn, focus, isAlive, markRemoved, showError,
} from './session.js';

let healthToken = 0;
let listToken = 0;
let navToken = 0;
let dropFromQueue = () => {};

/** Let the read queue hear when a document is forgotten. */
export function onForget(fn) {
  dropFromQueue = fn;
}

export async function loadHealth() {
  healthToken += 1;
  const token = healthToken;
  try {
    const health = await callJson(fetchFn, 'GET', paths.health(), {
      isCurrent: () => token === healthToken,
    });
    if (health !== STALE) setState({ health, healthError: null });
  } catch (err) {
    if (token === healthToken) setState({ healthError: errorMessage(err, { action: 'health' }) });
  }
}

/** Fetch the document list (oldest first); returns it, or STALE when a newer call was made. */
export async function loadList() {
  listToken += 1;
  const token = listToken;
  const list = await callJson(fetchFn, 'GET', paths.documents(), {
    isCurrent: () => token === listToken,
  });
  if (list !== STALE) setState({ list });
  return list;
}

/** Forget a document everywhere in this tab: queue, cache, list and edit box. */
export function forget(docId) {
  markRemoved(docId);
  dropFromQueue(docId);
  setState((s) => {
    const { [docId]: _gone, ...docs } = s.docs;
    return {
      docs,
      list: s.list.filter((item) => item.document_id !== docId),
      draft: s.draft && s.draft.docId === docId ? null : s.draft,
    };
  });
}

/** The document is no longer in folio: say so, forget it and leave it if it is open. */
export function handleGone(docId) {
  const state = getState();
  const name = state.docs[docId]?.source.filename
    ?? state.list.find((item) => item.document_id === docId)?.filename ?? '';
  forget(docId);
  setState({ alert: [msg.docGone(name)] });
  if (getState().docId === docId) setState({ screen: 'start', docId: null, lineId: null });
}

/** Errors that mean the document is gone are handled the same way everywhere. */
export function reportError(err, docId, ctx) {
  if (err instanceof ApiError && err.kind === 'http' && err.status === 404 && docId) {
    handleGone(docId);
    return;
  }
  showError(err, ctx);
}

/** Reload the image of every page whose read state or size changed (read in another window). */
function bumpChangedPages(readCounts, before, after) {
  if (!before) return readCounts;
  const changed = after.pages.filter((p, i) => {
    const old = before.pages[i];
    return !old || old.status !== p.status || old.width !== p.width || old.height !== p.height;
  });
  const bumped = changed.map((p) => {
    const key = `${after.id}:${p.index}`;
    return [key, (readCounts[key] ?? 0) + 1];
  });
  return { ...readCounts, ...Object.fromEntries(bumped) };
}

/** Start a move to another place; a document arriving for an earlier move is then dropped. */
function navigate() {
  navToken += 1;
  return navToken;
}

async function fetchDoc(docId, token) {
  const doc = await callJson(fetchFn, 'GET', paths.exported(docId, 'json'), {
    isCurrent: () => token === navToken && isAlive(docId),
  });
  if (doc === STALE) return STALE;
  setState((s) => ({
    docs: { ...s.docs, [docId]: doc },
    readCounts: bumpChangedPages(s.readCounts, s.docs[docId], doc),
  }));
  return doc;
}

/**
 * A read can change the document's engine (the reader and its confidence check), which the
 * read reply does not carry: fetch the document and take its engine only.
 */
let engineTokens = {};

export async function refreshEngine(docId) {
  const token = (engineTokens[docId] ?? 0) + 1;
  engineTokens = { ...engineTokens, [docId]: token };
  let fresh;
  try {
    fresh = await callJson(fetchFn, 'GET', paths.exported(docId, 'json'), {
      isCurrent: () => isAlive(docId) && engineTokens[docId] === token,
    });
  } catch (err) {
    reportError(err, docId, { action: 'open', name: docId });
    return;
  }
  if (fresh === STALE) return;
  setState((s) => (s.docs[docId] ? { docs: { ...s.docs, [docId]: { ...s.docs[docId], engine: fresh.engine } } } : {}));
}

function showDoc(docId, page, lineId) {
  const doc = getState().docs[docId];
  const n = clampPage(doc, page);
  const p = doc.pages[n - 1];
  const keep = lineId && p.lines.some((l) => l.id === lineId) ? lineId : defaultLineId(p);
  const state = getState();
  const same = state.docId === docId;
  setState({
    screen: 'doc', docId, page: n, lineId: keep,
    status: same ? state.status : '',
    draft: same ? state.draft : null,
  });
}

/** Open a document (fetching it when this tab has no copy) at a page and line. */
export async function openDoc(docId, { page = 1, lineId = null, refetch = false } = {}) {
  if (getState().docId !== docId && blockedByDraft()) return false;
  const token = navigate();
  if (refetch || !getState().docs[docId]) {
    try {
      if ((await fetchDoc(docId, token)) === STALE) return false;
    } catch (err) {
      if (token === navToken) reportError(err, docId, { action: 'open', name: docId });
      return false;
    }
    // The user may have moved, or typed in the edit box, while the document was on its way.
    if (token !== navToken) return false;
    if (getState().docId !== docId && blockedByDraft()) return false;
  }
  // A refresh of the open document keeps the place the user is at now.
  const here = refetch && getState().docId === docId ? getState() : { page, lineId };
  showDoc(docId, here.page, here.lineId);
  return true;
}

/** Fetch the open document again (the "Refresh" notice), keeping the place. */
export function refreshDoc() {
  const { docId, page, lineId } = getState();
  if (docId) openDoc(docId, { page, lineId, refetch: true });
}

export function goPage(n) {
  const state = getState();
  const doc = currentDoc(state);
  if (!doc || n < 1 || n > doc.pages.length || n === state.page) return;
  if (blockedByDraft()) return;
  setState({ page: n, lineId: defaultLineId(doc.pages[n - 1]), draft: null });
}

/** Leave the open document for the start screen, dropping any document still on its way. */
export function closeDoc() {
  navigate();
  setState({ screen: 'start', docId: null, lineId: null, draft: null });
}

export async function goStart() {
  if (blockedByDraft()) return;
  closeDoc();
  focus('start-title');
  try {
    await loadList();
  } catch (err) {
    showError(err, { action: 'list' });
  }
}
