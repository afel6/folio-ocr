// Why a page image failed to load. An image error carries no status, so folio is asked once
// more for the same image; only its "every reader is busy" reply changes the page notice.
import { callEmpty } from './api.js';
import { ApiError, isReadersBusy } from './errors.js';
import { IMAGE_BUSY } from './logic.js';
import { setState } from './store.js';
import { fetchFn, isAlive } from './session.js';

// The newest ask per page; a reply to an older one changes nothing.
let latest = {};

async function askedBusy(url) {
  try {
    await callEmpty(fetchFn, 'GET', url);
    return false;
  } catch (err) {
    if (!(err instanceof ApiError)) throw err;
    return isReadersBusy(err);
  }
}

/** Mark the image of page `n` as failed, then ask folio once why `url` failed. */
export async function onImageError(docId, n, url) {
  const key = `${docId}:${n}`;
  const ask = Symbol(key);
  latest = { ...latest, [key]: ask };
  setState((s) => ({ imageFailed: { ...s.imageFailed, [key]: true } }));
  const busy = await askedBusy(url);
  if (!busy || latest[key] !== ask || !isAlive(docId)) return;
  setState((s) => (s.imageFailed[key] === true ? { imageFailed: { ...s.imageFailed, [key]: IMAGE_BUSY } } : {}));
}
