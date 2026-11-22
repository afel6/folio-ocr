// folio review screen: wires the state to every part of the page and starts it.
import { $ } from './dom.js';
import { STALE } from './api.js';
import { msg } from './copy.js';
import { alertOnDrawFailure, drawParts, parseHash } from './logic.js';
import { getState, setState, subscribe } from './store.js';
import { draftChanged, showError, writeHash } from './session.js';
import { goPage, goStart, loadHealth, loadList, openDoc, refreshDoc } from './docs.js';
import { readPage, resumeReading, stopReading } from './reads.js';
import { onImageError } from './images.js';
import { deleteDoc, exportDoc, upload } from './files.js';
import { cancelEdit, confirm, draftInput, next, openEditor, save, selectLine } from './lines.js';
import * as chrome from './ui-chrome.js';
import * as start from './ui-start.js';
import * as pageView from './ui-page.js';
import * as review from './ui-review.js';
import * as dialogs from './ui-dialogs.js';

const parts = [chrome, start, pageView, review, dialogs];
const TICK_MS = 1000;
let ticker = null;

function runNotice(key) {
  const { docId, page } = getState();
  if (key === 'readThis' || key === 'readAgain') readPage(docId, page);
  else if (key === 'refresh') refreshDoc();
  else if (key === 'tryAgain') {
    setState({ alert: null });
    resumeReading();
  }
}

const actions = {
  goStart,
  openDoc: (id) => openDoc(id),
  goPage,
  selectLine,
  next: (dir) => next(dir),
  confirm,
  openEditor,
  draftInput,
  save,
  cancelEdit,
  runNotice,
  stopReading,
  readPage: (n) => readPage(getState().docId, n),
  exportDoc,
  askDelete: () => setState({ dialog: 'delete' }),
  deleteDoc,
  closeDialog: () => setState({ dialog: null }),
  openReaders: () => {
    setState({ dialog: 'readers' });
    loadHealth();
  },
  upload,
  dismissAlert: () => {
    setState({ alert: null });
    $(getState().screen === 'doc' ? 'doc-title' : 'start-title').focus();
  },
  imageError: onImageError,
};

let renderFailed = false;

/** Redraw every part; a drawing failure shows the error message once, through the state. */
function render(state) {
  const failed = drawParts(parts, state);
  if (alertOnDrawFailure(failed, renderFailed)) queueMicrotask(() => setState({ alert: msg.generic(null) }));
  renderFailed = failed;
  writeHash(state);
  keepTicking(state);
}

/** While a page is read, redraw each second so the elapsed time moves. */
function keepTicking(state) {
  if (state.inFlight && !ticker) ticker = setInterval(() => render(getState()), TICK_MS);
  if (!state.inFlight && ticker) {
    clearInterval(ticker);
    ticker = null;
  }
}

/** Reopen the place named in the URL hash, when its document is still in folio. */
async function restore(place) {
  let list;
  try {
    list = await loadList();
  } catch (err) {
    showError(err, { action: 'list' });
    return;
  }
  if (place && list !== STALE && list.some((item) => item.document_id === place.doc)) {
    await openDoc(place.doc, { page: place.page, lineId: place.line });
  }
}

function boot() {
  const place = parseHash(window.location.hash);
  for (const part of parts) part.init(actions);
  subscribe(render);
  render(getState());
  window.addEventListener('beforeunload', (event) => {
    if (draftChanged()) event.preventDefault();
  });
  loadHealth();
  restore(place);
}

boot();
