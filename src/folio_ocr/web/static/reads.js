// Reading pages: the queue that sends one read at a time, and what a finished read changes.
import { msg } from './copy.js';
import { PAGE_DONE, mergeReadPage, pageNotices, replacePage, toCheck } from './logic.js';
import { createReadQueue } from './queue.js';
import { getState, setState } from './store.js';
import { defaultLineId, fetchFn, isAlive, savedFor, stamp } from './session.js';
import { loadHealth, onForget, refreshEngine, reportError } from './docs.js';

const keyOf = (docId, n) => `${docId}:${n}`;
let held = { docId: null, note: '' };

/** Keep a note (the "added" line) in front of the reading status of that document's next read. */
export function holdNote(docId, note) {
  held = { docId, note };
}

function onStart(item) {
  const state = getState();
  const doc = state.docs[item.docId];
  if (state.docId === item.docId && doc) {
    const note = held.docId === item.docId ? held.note : '';
    setState({ status: [note, msg.readingStatus(item.page, doc.pages.length)].filter(Boolean).join(' ') });
    held = { docId: null, note: '' };
  }
}

/** Notes about a page, leaving out the "all lines checked" notice. */
function noticeCount(state, docId, page) {
  const key = keyOf(docId, page.index);
  return pageNotices(page, {
    queued: false,
    readingHere: false,
    imageFailed: Boolean(state.imageFailed[key]),
    modelConfigured: Boolean(state.health?.model?.configured),
  }).filter((notice) => notice !== PAGE_DONE).length;
}

function onDone(item, page, readSeq) {
  const merged = mergeReadPage(page, savedFor(item.docId), readSeq);
  const key = keyOf(item.docId, item.page);
  setState((s) => {
    const doc = s.docs[item.docId];
    if (!doc) return {};
    const { [key]: _failed, ...imageFailed } = s.imageFailed;
    const viewing = s.docId === item.docId && s.page === item.page;
    return {
      docs: { ...s.docs, [item.docId]: replacePage(doc, merged) },
      readCounts: { ...s.readCounts, [key]: (s.readCounts[key] ?? 0) + 1 },
      imageFailed,
      lineId: viewing && !merged.lines.some((l) => l.id === s.lineId) ? defaultLineId(merged) : s.lineId,
    };
  });
  const state = getState();
  if (state.docId === item.docId) {
    const notes = noticeCount(state, item.docId, merged) > 0;
    setState({ status: msg.readDone(item.page, merged.lines.length, toCheck(merged.lines), notes) });
  }
  loadHealth();
  refreshEngine(item.docId);
}

function onError(item, err) {
  const doc = getState().docs[item.docId];
  reportError(err, item.docId, {
    action: 'read', page: item.page, name: doc?.source.filename ?? '',
  });
  loadHealth();
}

const queue = createReadQueue({
  fetchFn,
  isAlive,
  stamp,
  onChange: (snapshot) => setState(snapshot),
  onStart,
  onDone,
  onError,
});

onForget((docId) => queue.dropDoc(docId));

/** Queue every page of a newly added document (all unread), in page order. */
export function queuePages(docId, pageCount) {
  if (!isAlive(docId)) return;
  queue.enqueue(Array.from({ length: pageCount }, (_, i) => ({ docId, page: i + 1 })));
}

/** "Read this page" / "Read again": the page goes to the front of the queue. */
export function readPage(docId, n) {
  queue.front({ docId, page: n });
}

export const stopReading = () => queue.stop();
export const resumeReading = () => queue.resume();
