// Page column: page bar, reading progress, page image with line boxes, page notices.

import { $, h, s, attr, show, rich } from './dom.js';
import * as copy from './copy.js';
import * as logic from './logic.js';
import { paths } from './api.js';

const MARKERS = { needs_review: '!', edited: '✎', confirmed: '✓' };

// What was last drawn, so render only touches what changed.
let current = null;
let drawn = { doc: null, page: null, img: null, noticesKey: null, selected: null, overlayPage: null };

export function init(actions) {
  $('prev-page').addEventListener('click', () => current && actions.goPage(current.page - 1));
  $('next-page').addEventListener('click', () => current && actions.goPage(current.page + 1));
  $('page-select').addEventListener('change', (e) => actions.goPage(Number(e.target.value)));
  $('read-page').addEventListener('click', () => current && actions.readPage(current.page));
  $('stop-reading').addEventListener('click', () => actions.stopReading());
  $('page-img').addEventListener('error', () => {
    if (current) actions.imageError(current.docId, current.page);
  });
  $('overlay').addEventListener('click', (e) => {
    const box = e.target.closest('g.box');
    if (box) actions.selectLine(box.dataset.lineId, { focusList: false });
  });
  $('notices').addEventListener('click', (e) => {
    const button = e.target.closest('button[data-action]');
    if (button) actions.runNotice(button.dataset.action);
  });
}

export function render(state) {
  const doc = state.docs[state.docId];
  const page = doc?.pages[state.page - 1];
  if (!doc || !page) return;
  const n = state.page;
  const key = `${state.docId}:${n}`;
  const readingHere = state.inFlight?.docId === state.docId && state.inFlight.page === n;
  const queued = logic.isQueued(state, state.docId, n);
  current = { docId: state.docId, page: n };

  $('page-heading').textContent = `Page ${n} of ${doc.pages.length}`;
  attr($('page'), 'aria-busy', readingHere ? 'true' : null);
  renderPageBar(state, doc, page, { readingHere, queued });
  renderProgress(state);
  renderImage(state, doc, page, key);
  renderOverlay(page, state.lineId);
  renderNotices(state, page, key, { readingHere, queued });
  drawn = { ...drawn, doc, page };
}

function renderPageBar(state, doc, page, { readingHere, queued }) {
  const n = state.page;
  const total = doc.pages.length;
  $('prev-page').disabled = n <= 1;
  $('next-page').disabled = n >= total;
  if (drawn.doc !== doc) {
    $('page-select').replaceChildren(
      ...doc.pages.map((p, i) =>
        h('option', { value: i + 1 }, copy.msg.pageOption(i + 1, total, logic.toCheck(p.lines))),
      ),
    );
  }
  $('page-select').value = String(n);
  const button = $('read-page');
  const label = logic.readButtonLabel(page, { queued, readingHere });
  show(button, label !== null);
  if (label !== null && button.textContent !== label) button.textContent = label;
  show($('stop-reading'), state.queue.length > 0);
}

function renderProgress(state) {
  const flight = state.inFlight;
  const active = flight?.docId === state.docId;
  show($('progress-wrap'), active);
  if (!active) return;
  attr($('progress'), 'aria-label', copy.msg.progressLabel(flight.page));
  $('progress').removeAttribute('value');
  $('progress-label').textContent = copy.msg.readingBar(flight.page);
  $('elapsed').textContent = copy.elapsed(performance.now() - flight.startedAt);
}

function hasSize(page) {
  return page.width > 0 && page.height > 0;
}

function renderImage(state, doc, page, key) {
  const img = $('page-img');
  const url = paths.image(state.docId, state.page, state.readCounts[key] || 0);
  if (drawn.img !== url) {
    img.setAttribute('src', url);
    drawn = { ...drawn, img: url };
  }
  const file = doc.source.filename;
  if (hasSize(page)) {
    img.setAttribute('alt', `Page ${state.page} of ${file}, straightened. Line boxes are drawn on it; review the lines in the list.`);
    img.setAttribute('width', String(page.width));
    img.setAttribute('height', String(page.height));
  } else {
    img.setAttribute('alt', `Page ${state.page} of ${file}.`);
    img.removeAttribute('width');
    img.removeAttribute('height');
  }
}

function rectAttrs(cls, [x0, y0, x1, y1]) {
  return { class: cls, x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

function lineBox(line, W, H) {
  const [x0, y0, x1] = line.box;
  const children = ['halo', 'under', 'edge'].map((cls) => s('rect', rectAttrs(cls, line.box)));
  const glyph = MARKERS[line.status];
  if (glyph) {
    const fs = Math.max(12, Math.round(H / 60));
    let x = x1 + 0.3 * fs;
    if (x + fs > W) x = x0 - 1.1 * fs;
    children.push(s('text', { class: 'marker', x, y: y0 + fs, 'font-size': fs }, glyph));
  }
  return s('g', { class: 'box', 'data-line-id': line.id, 'data-status': line.status }, ...children);
}

function hatchDefs() {
  return s('defs', {},
    s('pattern', { id: 'hatch', patternUnits: 'userSpaceOnUse', width: 16, height: 16, patternTransform: 'rotate(45)' },
      s('line', { class: 'hatch-line', x1: 0, y1: 0, x2: 0, y2: 16 })));
}

function buildOverlay(page) {
  return [
    hatchDefs(),
    ...(page.unread_regions ?? []).map((box) => s('rect', { ...rectAttrs('unread', box), fill: 'url(#hatch)' })),
    ...(page.lines ?? []).map((line) => lineBox(line, page.width, page.height)),
  ];
}

function renderOverlay(page, lineId) {
  const svg = $('overlay');
  const visible = hasSize(page) && ((page.lines?.length ?? 0) > 0 || (page.unread_regions?.length ?? 0) > 0);
  show(svg, visible);
  if (!visible) {
    if (drawn.overlayPage !== null) svg.replaceChildren();
    drawn = { ...drawn, overlayPage: null };
    return;
  }
  let rebuilt = false;
  if (drawn.overlayPage !== page) {
    svg.setAttribute('viewBox', `0 0 ${page.width} ${page.height}`);
    svg.replaceChildren(...buildOverlay(page));
    drawn = { ...drawn, overlayPage: page };
    rebuilt = true;
  }
  if (rebuilt || drawn.selected !== lineId) markSelected(svg, lineId, drawn.selected !== lineId);
}

function markSelected(svg, lineId, scroll) {
  let target = null;
  for (const g of svg.querySelectorAll('g.box')) {
    const on = g.dataset.lineId === lineId;
    g.classList.toggle('selected', on);
    if (on) target = g;
  }
  drawn = { ...drawn, selected: lineId };
  if (target && scroll) {
    const smooth = matchMedia('(prefers-reduced-motion: no-preference)').matches;
    target.scrollIntoView({ block: 'nearest', behavior: smooth ? 'smooth' : 'auto' });
  }
}

function renderNotices(state, page, key, { readingHere, queued }) {
  const notices = logic.pageNotices(page, {
    queued,
    readingHere,
    imageFailed: !!state.imageFailed[key],
    modelConfigured: !!state.health?.model?.configured,
  });
  const signature = JSON.stringify(notices);
  if (signature === drawn.noticesKey) return;
  $('notices').replaceChildren(...notices.map((segments) => h('p', { class: 'notice' }, rich(segments))));
  drawn = { ...drawn, noticesKey: signature };
}
