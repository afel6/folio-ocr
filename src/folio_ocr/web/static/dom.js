// Small DOM helpers. Text always goes in through text nodes, never through markup.

const SVG_NS = 'http://www.w3.org/2000/svg';

export const $ = (id) => document.getElementById(id);

function applyAttrs(node, attrs) {
  for (const [name, value] of Object.entries(attrs ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    node.setAttribute(name, value === true ? '' : String(value));
  }
}

function append(node, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
}

/** Create an HTML element; string children become text nodes. */
export function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  applyAttrs(node, attrs);
  append(node, children);
  return node;
}

/** Create an SVG element with attributes. */
export function s(tag, attrs, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  applyAttrs(node, attrs);
  append(node, children);
  return node;
}

/** Set an element's attribute, or remove it when the value is null. */
export function attr(node, name, value) {
  if (value === null || value === undefined) node.removeAttribute(name);
  else node.setAttribute(name, String(value));
}

/** Set an element's text only when it differs, so unchanged text is not redrawn. */
export function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

/** Show or hide an element with the hidden attribute. */
export function show(node, visible) {
  node.hidden = !visible;
}

/**
 * Build rich copy from segments: a string is text, {code} is <code translate="no">,
 * {button, label} is a <button type="button" data-action=button>.
 */
export function rich(segments) {
  const frag = document.createDocumentFragment();
  for (const seg of segments) {
    if (typeof seg === 'string') frag.append(document.createTextNode(seg));
    else if (seg.code !== undefined) frag.append(h('code', { translate: 'no' }, seg.code));
    else if (seg.button !== undefined) {
      frag.append(h('button', { type: 'button', 'data-action': seg.button }, seg.label));
    }
  }
  return frag;
}
