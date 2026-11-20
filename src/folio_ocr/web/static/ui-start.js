// Start screen: upload form, drop zone and the note about held documents.
import { $, rich, show, attr } from './dom.js';
import * as logic from './logic.js';

let acts = null;
let uploading = false;
let lastList;

function selectedLanguages() {
  return document.querySelector('#upload input[name="languages"]:checked')?.value;
}

function initDrop() {
  const drop = $('drop');
  const over = (event) => {
    event.preventDefault();
    drop.classList.add('over');
  };
  drop.addEventListener('dragenter', over);
  drop.addEventListener('dragover', over);
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    drop.classList.remove('over');
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) $('file').files = files;
  });
}

export function init(actions) {
  acts = actions;
  uploading = false;
  lastList = undefined;
  $('upload').addEventListener('submit', (event) => {
    event.preventDefault();
    if (uploading) return;
    const file = $('file').files[0];
    if (file) acts.upload(file, selectedLanguages());
  });
  initDrop();
}

export function render(state) {
  uploading = state.uploading;
  attr($('add-read'), 'aria-disabled', state.uploading ? 'true' : null);
  if (state.list === lastList) return;
  lastList = state.list;
  const note = logic.heldNote(state.list);
  show($('held-note'), note !== null);
  $('held-note').replaceChildren(note ? rich(note) : '');
}
