// Reviewing lines: selecting, moving to the next line to check, editing, saving, confirming.
import { $ } from './dom.js';
import { STALE, callJson, paths } from './api.js';
import { msg } from './copy.js';
import { nextToCheck, normaliseNewlines, notReadPages, replaceLine, textToSave } from './logic.js';
import { getState, setState } from './store.js';
import {
  blockedByDraft, currentDoc, currentLine, currentPage, draftChanged, fetchFn, focus, lineK,
  recordSaved, trackSave,
} from './session.js';
import { reportError } from './docs.js';

export function selectLine(lineId, { focusList = false } = {}) {
  const state = getState();
  if (lineId !== state.lineId) {
    if (blockedByDraft()) return;
    setState({ lineId, draft: null });
  }
  if (focusList) focus('lines');
}

/** Go to the next (dir 1) or previous (dir -1) line to check, opening its page when needed. */
export function next(dir, prefix = '') {
  const state = getState();
  const doc = currentDoc(state);
  if (!doc) return;
  const hit = nextToCheck(doc, state.page - 1, state.lineId, dir);
  if (!hit) {
    setState({ status: `${prefix}${msg.nothingLeft(notReadPages(doc).length)}` });
    return;
  }
  if (hit.lineId !== state.lineId && blockedByDraft()) return;
  setState({ page: hit.page + 1, lineId: hit.lineId, draft: null, ...(prefix ? { status: prefix.trim() } : {}) });
}

const isDraftFor = (draft, docId, lineId) => Boolean(draft) && draft.docId === docId && draft.lineId === lineId;

export function openEditor() {
  const state = getState();
  const line = currentLine(state);
  if (!line) return;
  if (!isDraftFor(state.draft, state.docId, line.id)) {
    setState({ draft: { docId: state.docId, lineId: line.id, value: normaliseNewlines(line.final_text) } });
  }
  const editor = $('editor');
  editor.focus();
  editor.setSelectionRange(editor.value.length, editor.value.length);
}

export function draftInput(value) {
  setState((s) => (s.draft ? { draft: { ...s.draft, value } } : {}));
}

export function cancelEdit() {
  setState({ draft: null });
  focus('lines');
}

/** PUT a line's final text; on success store the returned line and go to the next line to check. */
async function putLine(docId, line, text) {
  const page = currentPage();
  const k = lineK(page, line);
  let result;
  try {
    result = await trackSave(callJson(fetchFn, 'PUT', paths.line(docId, line.id), {
      json: { final_text: text },
      isCurrent: () => Boolean(getState().docs[docId]),
    }));
  } catch (err) {
    reportError(err, docId, { action: 'save', line: k, name: getState().docs[docId]?.source.filename });
    return;
  }
  if (result === STALE) return;
  recordSaved(docId, result);
  const stillHere = getState().docId === docId && getState().lineId === line.id;
  // Text typed while the save was on its way stays in the box, and the box keeps focus.
  const draft = getState().draft;
  const keepDraft = isDraftFor(draft, docId, line.id) && textToSave(draft.value, text) !== text;
  setState((s) => ({
    docs: { ...s.docs, [docId]: { ...replaceLine(s.docs[docId], result), has_edits: true } },
    draft: isDraftFor(s.draft, docId, line.id) && !keepDraft ? null : s.draft,
  }));
  const done = result.status === 'confirmed' ? msg.confirmed(k) : msg.saved(k);
  if (!stillHere || keepDraft) {
    setState({ status: done });
    return;
  }
  next(1, `${done} `);
  focus('lines');
}

/** Save the edit box: an unchanged box sends the stored text byte for byte. */
export function save() {
  const state = getState();
  const line = currentLine(state);
  if (!line || !state.draft || state.draft.lineId !== line.id) return;
  putLine(state.docId, line, textToSave(state.draft.value, line.final_text));
}

/** Confirm sends the stored final text unchanged; it works on any status. */
export function confirm() {
  const state = getState();
  const line = currentLine(state);
  if (!line) return;
  if (draftChanged(state)) {
    blockedByDraft();
    return;
  }
  if (state.draft) setState({ draft: null });
  putLine(state.docId, line, line.final_text);
}
