// Page reads, one at a time. No DOM. The server allows a single read at a time, so every
// page waits its turn here, across documents.
import { STALE, callJson, paths } from './api.js';

const same = (a, b) => a.docId === b.docId && a.page === b.page;
const copy = (item) => ({ docId: item.docId, page: item.page });

export function createReadQueue({ fetchFn, isAlive, stamp, onChange, onStart, onDone, onError }) {
  let queue = [];
  let inFlight = null;
  let paused = false;

  const snapshot = () => ({
    queue: queue.map(copy),
    inFlight: inFlight ? { ...inFlight } : null,
    paused,
  });
  const changed = () => onChange(snapshot());
  const known = (item) => queue.some((q) => same(q, item)) || (inFlight && same(inFlight, item));

  function dropDoc(docId) {
    queue = queue.filter((q) => q.docId !== docId);
    changed();
  }

  /** Decide what an error means for the rest of the queue. */
  function afterError(item, err) {
    const status = err?.kind === 'http' ? err.status : null;
    if (status === 409) {
      queue = [copy(item), ...queue];
      paused = true;
    } else if (status === 404) {
      queue = queue.filter((q) => q.docId !== item.docId);
    } else if (status !== 400) {
      queue = [];
    }
  }

  async function run(item) {
    queue = queue.slice(1);
    const readSeq = stamp();
    inFlight = { ...copy(item), startedAt: performance.now() };
    onStart(copy(item));
    changed();
    let page;
    let failure = null;
    try {
      page = await callJson(fetchFn, 'POST', paths.read(item.docId, item.page), {
        isCurrent: () => isAlive(item.docId),
      });
    } catch (err) {
      failure = err;
    }
    inFlight = null;
    if (failure) {
      onError(copy(item), failure);
      afterError(item, failure);
    } else if (page !== STALE) {
      onDone(copy(item), page, readSeq);
    }
    changed();
    pump();
  }

  function pump() {
    if (inFlight || paused || queue.length === 0) return;
    run(queue[0]);
  }

  function enqueue(items) {
    const fresh = [];
    for (const it of items) {
      if (!known(it) && !fresh.some((f) => same(f, it))) fresh.push(copy(it));
    }
    if (fresh.length === 0) return;
    queue = [...queue, ...fresh];
    changed();
    pump();
  }

  function front(item) {
    if (inFlight && same(inFlight, item)) return;
    queue = [copy(item), ...queue.filter((q) => !same(q, item))];
    paused = false;
    changed();
    pump();
  }

  function resume() {
    paused = false;
    changed();
    pump();
  }

  function stop() {
    queue = [];
    paused = false;
    changed();
  }

  return { enqueue, front, stop, dropDoc, resume, snapshot };
}
