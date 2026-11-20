// Page chrome: document navigation, readers button, banners, document header,
// message bar, window title and the skip link.
import { $, h, rich, setText, show } from './dom.js';
import { msg } from './copy.js';
import * as logic from './logic.js';
import { currentDoc, currentPage } from './session.js';

let acts = null;
let lastAlert;
let lastListKey = null;

function cachedDoc(state) {
  return state.screen === 'doc' ? currentDoc(state) : null;
}

function docEntries(state) {
  return state.list.map((item) => {
    const id = item.document_id;
    return {
      id,
      name: item.filename,
      count: logic.countText(item, state.docs[id] ?? null),
      current: state.screen === 'doc' && id === state.docId,
    };
  });
}

function renderDocList(state) {
  const entries = docEntries(state);
  const key = JSON.stringify(entries);
  if (key === lastListKey) return;
  lastListKey = key;
  $('doc-list').replaceChildren(
    ...entries.map((e) => {
      const btn = h(
        'button',
        { type: 'button', class: 'doc-btn', 'data-doc-id': e.id, 'aria-current': e.current ? 'page' : null },
        h('span', { class: 'doc-name' }, e.name),
        h('span', { class: 'doc-count' }, e.count),
      );
      return h('li', null, btn);
    }),
  );
}

function renderReaders(state) {
  const r = logic.readersState(state.healthError ? null : state.health);
  const text = { ready: 'Readers: ready', check: 'Readers: check', unknown: 'Readers: unknown' }[r];
  setText($('readers-text'), text);
  show($('readers-mark'), r === 'check');
}

function renderBanners(state) {
  const model = state.health?.model;
  const leaves = model?.leaves_machine === true;
  show($('banner-model'), leaves);
  if (leaves) setText($('banner-model'), msg.modelBanner(model.host));
  show($('banner-remote'), logic.isRemoteHost(location.hostname));
}

function renderDocHead(state, doc) {
  const open = doc !== null;
  show($('doc-head'), open);
  show($('work'), open);
  if (!open) return;
  setText($('doc-title'), doc.source.filename);
  setText($('doc-meta'), msg.docMeta(doc.languages, doc.pages.length, doc.created_at));
  setText($('export-summary'), logic.exportSummary(doc, logic.readingPages(state, doc)));
  show($('uncalibrated'), logic.everyLineFlagged(doc));
}

function renderMessages(state) {
  setText($('status'), state.status);
  if (state.alert !== lastAlert) {
    lastAlert = state.alert;
    $('alert').replaceChildren(state.alert ? rich(state.alert) : '');
  }
  show($('dismiss'), state.alert !== null);
}

function renderTitle(state, doc) {
  document.title = doc ? msg.pageTitle(state.page, doc.pages.length, doc.source.filename) : 'folio';
  const lines = doc ? currentPage(state)?.lines : null;
  show($('skip'), Boolean(lines && lines.length > 0));
}

export function init(actions) {
  acts = actions;
  lastAlert = undefined;
  lastListKey = null;
  $('doc-list').addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-doc-id]');
    if (btn) acts.openDoc(btn.getAttribute('data-doc-id'));
  });
  $('skip').addEventListener('click', (event) => {
    event.preventDefault(); // keep the document's place in the URL hash
    $('lines').focus();
  });
  $('add-doc').addEventListener('click', () => acts.goStart());
  $('readers').addEventListener('click', () => acts.openReaders());
  $('dismiss').addEventListener('click', () => acts.dismissAlert());
  $('delete-doc').addEventListener('click', () => acts.askDelete());
  for (const btn of document.querySelectorAll('#doc-head button[data-format]')) {
    btn.addEventListener('click', () => acts.exportDoc(btn.getAttribute('data-format')));
  }
  $('alert').addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-action]');
    if (btn) acts.runNotice(btn.getAttribute('data-action'));
  });
}

export function render(state) {
  const doc = cachedDoc(state);
  renderDocList(state);
  renderReaders(state);
  renderBanners(state);
  show($('start'), state.screen === 'start');
  renderDocHead(state, doc);
  renderMessages(state);
  renderTitle(state, doc);
}
