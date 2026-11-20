// Review column: the line list (a listbox with keyboard control) and the detail panel
// for the selected line, including the edit box.
import { $, h, s, attr, setText, show } from './dom.js';
import { fmtNum, plural, statusLabel, flagCopy } from './copy.js';
import { closeupBox, editorKeyAction, lineAttrs, lineNumber, listKeyAction, toCheck } from './logic.js';
import { currentPage } from './session.js';
import { paths } from './api.js';

// What the last render drew, so unchanged parts of the page are left alone.
let last = { page: null, lines: [], lineId: null, line: null, closeKey: null };

// ---------- small helpers ----------

function setLineText(node, line, text) {
  const { dir, lang } = lineAttrs(line);
  attr(node, 'dir', dir);
  attr(node, 'lang', lang);
  setText(node, text);
}

function statusSpan(status) {
  const { label, glyph } = statusLabel(status);
  return h(
    'span',
    { class: 'status-label', 'data-status': status },
    glyph ? h('span', { class: 'glyph', 'aria-hidden': 'true' }, glyph) : null,
    label,
  );
}

function isEditing(state, line) {
  const d = state.draft;
  return Boolean(d && line && d.docId === state.docId && d.lineId === line.id);
}

// ---------- line list ----------

const hidden = (text) => h('span', { class: 'visually-hidden' }, text);

/** A list row; its accessible name, read from its content, is "Line 5, To check, Unsure reading: <text>". */
function buildOption(line, index) {
  const k = lineNumber(line, index);
  const codes = line.flags;
  const meta = h('span', { class: 'meta' }, statusSpan(line.status));
  if (codes.length > 0) {
    meta.append(hidden(', '), h('span', { class: 'flag-first' }, flagCopy(codes[0]).label));
    if (codes.length > 1) {
      meta.append(' ', h('span', { class: 'more', 'aria-hidden': 'true' }, `+${fmtNum(codes.length - 1)} more`));
    }
  }
  meta.append(hidden(': '));
  const text = h('span', { class: 'line-text' });
  setLineText(text, line, line.final_text);
  return h(
    'div',
    {
      role: 'option',
      class: 'option',
      id: `opt-${line.id}`,
      'data-line-id': line.id,
      'data-status': line.status,
      'aria-selected': 'false',
    },
    h('span', { class: 'num' }, hidden('Line '), String(k), hidden(', ')),
    meta,
    text,
  );
}

function syncSelection(list, lineId) {
  let selected = null;
  for (const option of list.children) {
    const on = option.getAttribute('data-line-id') === lineId;
    const value = on ? 'true' : 'false';
    if (option.getAttribute('aria-selected') !== value) option.setAttribute('aria-selected', value);
    if (on) selected = option;
  }
  attr(list, 'aria-activedescendant', selected ? selected.id : null);
  return selected;
}

function renderList(state, page, lineId) {
  const list = $('lines');
  const lines = page?.lines ?? [];
  const has = lines.length > 0;
  show(list, has);
  show(document.querySelector('ul.keys'), has);
  if (!has) {
    list.replaceChildren();
    attr(list, 'aria-activedescendant', null);
    return;
  }
  attr(list, 'aria-label', `Lines on page ${state.page}`);
  const rebuilt = page !== last.page;
  if (rebuilt) list.replaceChildren(...lines.map(buildOption));
  const selected = syncSelection(list, lineId);
  if (selected && (rebuilt || lineId !== last.lineId)) selected.scrollIntoView({ block: 'nearest' });
}

// ---------- detail panel ----------

function renderCloseup(state, page, line, k) {
  const svg = $('closeup');
  const ok = page.width > 0 && page.height > 0;
  show(svg, ok);
  if (!ok) {
    last = { ...last, closeKey: null };
    return;
  }
  const version = state.readCounts?.[`${state.docId}:${state.page}`] || 0;
  const href = paths.image(state.docId, state.page, version);
  const box = closeupBox(page, line);
  const key = `${href}|${page.width}|${page.height}|${box.join(',')}`;
  attr(svg, 'aria-label', `Image of line ${k}`);
  if (key === last.closeKey) return;
  attr(svg, 'viewBox', box.join(' '));
  svg.replaceChildren(s('image', { href, width: page.width, height: page.height }));
  last = { ...last, closeKey: key };
}

function renderFlags(line) {
  const items = line.flags.map((code) => {
    const { label, sentence } = flagCopy(code);
    return h(
      'li',
      null,
      h('strong', { class: 'flag-label' }, `${label}:`),
      ' ',
      h('span', { class: 'flag-sentence' }, sentence),
    );
  });
  $('flags').replaceChildren(...items);
}

function renderStatus(line) {
  const node = $('detail-status');
  const { label, glyph } = statusLabel(line.status);
  attr(node, 'data-status', line.status);
  node.replaceChildren(
    ...(glyph ? [h('span', { class: 'glyph', 'aria-hidden': 'true' }, glyph)] : []),
    label,
  );
}

function renderEditor(state, line) {
  const editing = isEditing(state, line);
  const editor = $('editor');
  show(editor, editing);
  show($('edit-actions'), editing);
  show($('edit'), !editing);
  show($('detail-text'), !editing);
  if (!editing) return;
  const { dir, lang } = lineAttrs(line);
  attr(editor, 'dir', dir);
  attr(editor, 'lang', lang);
  attr(editor, 'placeholder', line.reader === 'none' && line.text === '' ? 'Type the text from the image…' : null);
  if (editor.value !== state.draft.value) editor.value = state.draft.value;
}

function renderDetailBody(state, page, line) {
  const k = lineNumber(line, page.lines.indexOf(line));
  setText($('detail-heading'), `Line ${k}`);
  renderCloseup(state, page, line, k);
  setLineText($('detail-text'), line, line.final_text);
  show($('read-as'), line.status === 'edited');
  if (line.status === 'edited') setLineText($('read-as-text'), line, line.text);
  show($('nobody-hint'), line.reader === 'none' && line.text === '');
  if (line !== last.line) {
    renderStatus(line);
    renderFlags(line);
  }
}

function renderDetail(state, page, line) {
  const detail = $('detail');
  show(detail, Boolean(line));
  if (!line) return;
  renderDetailBody(state, page, line);
  renderEditor(state, line);
}

// ---------- public ----------

export function render(state) {
  const page = currentPage(state);
  const lines = page?.lines ?? [];
  const line = lines.find((l) => l.id === state.lineId) ?? null;
  setText($('review-heading'), `Lines on page ${state.page}`);
  const counts = lines.length === 0
    ? ''
    : `${plural(lines.length, 'line', 'lines')}, ${fmtNum(toCheck(lines))} to check`;
  setText($('review-counts'), counts);
  renderList(state, page, line ? line.id : null);
  renderDetail(state, page, line);
  last = { ...last, page, lines, lineId: line ? line.id : null, line };
}

/** Run the action a key maps to, and keep the browser's own handling of the key only when none maps. */
function runKey(event, call, actions) {
  if (!call) return;
  event.preventDefault();
  const [name, ...args] = call;
  actions[name](...args);
}

export function init(actions) {
  $('lines').addEventListener('click', (event) => {
    const option = event.target.closest?.('[data-line-id]');
    if (option) actions.selectLine(option.getAttribute('data-line-id'), { focusList: true });
  });
  $('lines').addEventListener('keydown', (event) => runKey(event, listKeyAction(event, last.lines, last.lineId), actions));
  const editor = $('editor');
  editor.addEventListener('input', () => actions.draftInput(editor.value));
  editor.addEventListener('keydown', (event) => runKey(event, editorKeyAction(event), actions));
  $('save').addEventListener('click', () => actions.save());
  $('cancel').addEventListener('click', () => actions.cancelEdit());
  $('edit').addEventListener('click', () => actions.openEditor());
  $('confirm').addEventListener('click', () => actions.confirm());
  $('next-check').addEventListener('click', () => actions.next(1));
}
