// Adding, exporting and deleting documents.
import { STALE, callBlob, callEmpty, callJson, paths } from './api.js';
import { msg } from './copy.js';
import { $, h } from './dom.js';
import { docToCheck, evicted, nextDocId, partlyReadPages, stemOf } from './logic.js';
import { getState, setState } from './store.js';
import { currentDoc, fetchFn, focus, isAlive, pendingSaves, showError } from './session.js';
import { closeDoc, forget, goStart, loadList, openDoc, reportError } from './docs.js';
import { holdNote, queuePages } from './reads.js';

const DOWNLOAD_URL_LIFETIME_MS = 60_000;

async function listOrCurrent() {
  try {
    const list = await loadList();
    return list === STALE ? getState().list : list;
  } catch (err) {
    showError(err, { action: 'list' });
    return getState().list;
  }
}

/** Forget documents removed to make room for the upload; returns the notes naming them. */
async function evictedNotes(reply, before) {
  const after = await listOrCurrent();
  const gone = evicted(before, after).filter((item) => item.document_id !== reply.document_id);
  for (const item of gone) forget(item.document_id);
  return gone.map((item) => msg.evicted(item.filename));
}

/** Send a file to folio, open it at page 1 and queue its pages for reading. */
export async function upload(file, languages) {
  if (getState().uploading) return;
  const name = file?.name ?? '';
  setState({ uploading: true, status: msg.uploading(name), alert: null });
  const before = await listOrCurrent();
  const form = new FormData();
  form.append('file', file);
  form.append('languages', languages);
  let reply;
  try {
    reply = await callJson(fetchFn, 'POST', paths.documents(), { form });
  } catch (err) {
    setState({ uploading: false, status: '' });
    showError(err, { action: 'upload', file: name });
    return;
  }
  const notes = await evictedNotes(reply, before);
  setState({ uploading: false, status: '' });
  $('file').value = '';
  const opened = await openDoc(reply.document_id, { page: 1 });
  const filename = getState().docs[reply.document_id]?.source.filename ?? name;
  const added = [msg.added(filename, reply.page_count), ...notes].join(' ');
  setState({ status: added });
  holdNote(reply.document_id, added);
  if (opened) focus('doc-title');
  queuePages(reply.document_id, reply.page_count);
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = h('a', { href: url, download: filename, hidden: true });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), DOWNLOAD_URL_LIFETIME_MS);
}

/** Wait for pending saves, fetch the export and hand it to the browser as a download. */
export async function exportDoc(format) {
  const doc = currentDoc();
  if (!doc) return;
  const file = doc.source.filename;
  await Promise.allSettled(pendingSaves());
  // The export holds the document as it is now; later changes are not in the file.
  const held = getState().docs[doc.id] ?? doc;
  let reply;
  try {
    reply = await callBlob(fetchFn, paths.exported(doc.id, format), { isCurrent: () => isAlive(doc.id) });
  } catch (err) {
    reportError(err, doc.id, { action: 'export', file });
    return;
  }
  if (reply === STALE) return;
  const stem = stemOf(file);
  download(reply.blob, `${stem}.${format}`);
  setState({ status: msg.exported(stem, format, docToCheck(held), partlyReadPages(held)) });
}

export async function deleteDoc() {
  const doc = currentDoc();
  setState({ dialog: null });
  if (!doc) return;
  const file = doc.source.filename;
  try {
    await callEmpty(fetchFn, 'DELETE', paths.document(doc.id));
  } catch (err) {
    reportError(err, doc.id, { action: 'delete', file, name: file });
    return;
  }
  const next = nextDocId(getState().list, doc.id);
  const wasOpen = getState().docId === doc.id;
  forget(doc.id);
  if (!wasOpen) {
    setState({ status: msg.deleted(file) });
    return;
  }
  closeDoc();
  if (next) await openDoc(next, { page: 1 });
  else await goStart();
  setState({ status: msg.deleted(file) });
  focus('docs-heading');
}
