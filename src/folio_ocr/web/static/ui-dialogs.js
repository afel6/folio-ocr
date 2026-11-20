// Delete and Readers dialogs. Focus returns to the opener when one closes.
import { $, h, rich } from './dom.js';
import { msg } from './copy.js';
import * as logic from './logic.js';

let lastState = null;
let lastRowsKey = null;
const openers = { delete: null, readers: null };

function openDialog(name, dialog) {
  openers[name] = document.activeElement;
  dialog.showModal();
}

function syncDialog(state, name, dialog, fill) {
  const wanted = state.dialog === name;
  if (wanted && !dialog.open) {
    fill?.(state);
    openDialog(name, dialog);
  } else if (!wanted && dialog.open) {
    dialog.close();
  }
}

function fillDelete(state) {
  const doc = state.docs[state.docId];
  if (!doc) return;
  $('delete-title').textContent = msg.deleteTitle(doc.source.filename);
  $('delete-body').textContent = msg.deleteBody(doc.has_edits);
}

function renderReaderRows(state) {
  if (!$('readers-dialog').open) return;
  const rows = state.healthError
    ? [{ name: null, segments: state.healthError }]
    : logic.readersRows(state.health);
  const key = JSON.stringify(rows);
  if (key === lastRowsKey) return;
  lastRowsKey = key;
  const nodes = [];
  for (const row of rows) {
    if (row.name !== null) nodes.push(h('dt', null, row.name));
    const dd = h('dd');
    dd.append(rich(row.segments));
    nodes.push(dd);
  }
  $('readers-rows').replaceChildren(...nodes);
}

function wireClose(name, dialog) {
  dialog.addEventListener('close', () => {
    const opener = openers[name];
    openers[name] = null;
    const active = document.activeElement;
    const focusLost = active === document.body || dialog.contains(active);
    if (opener && opener.isConnected && focusLost) opener.focus();
    if (lastState && lastState.dialog === name) acts.closeDialog();
    lastRowsKey = null;
  });
}

let acts = null;

export function init(actions) {
  acts = actions;
  $('keep-doc').addEventListener('click', () => acts.closeDialog());
  $('confirm-delete').addEventListener('click', () => acts.deleteDoc());
  $('readers-close').addEventListener('click', () => acts.closeDialog());
  wireClose('delete', $('delete-dialog'));
  wireClose('readers', $('readers-dialog'));
}

export function render(state) {
  lastState = state;
  syncDialog(state, 'delete', $('delete-dialog'), fillDelete);
  syncDialog(state, 'readers', $('readers-dialog'));
  renderReaderRows(state);
}
