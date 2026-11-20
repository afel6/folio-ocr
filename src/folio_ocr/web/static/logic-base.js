// Small facts about lines, pages and documents. Pure; nothing is changed in place.

const SCRIPT_LANG = { arabic: 'ar', latin: 'en' };

/** Text direction and language for a line's text element. */
export function lineAttrs(line) {
  const dir = line.direction === 'ltr' || line.direction === 'rtl' ? line.direction : 'auto';
  let lang = SCRIPT_LANG[line.script] ?? null;
  if (line.script === 'mixed') lang = dir === 'rtl' ? 'ar' : 'en';
  return { dir, lang };
}

/** The line's number k from its id `p{n}-l{k}`, else its list position. */
export function lineNumber(line, index) {
  const match = /^p\d+-l(\d+)$/.exec(line.id ?? '');
  return match ? Number(match[1]) : index + 1;
}

export const toCheck = (lines) => lines.filter((l) => l.status === 'needs_review').length;
export const docToCheck = (doc) => doc.pages.reduce((sum, p) => sum + toCheck(p.lines), 0);
export const notReadPages = (doc) => doc.pages.filter((p) => p.status !== 'read');
/** Pages where some areas weren't read, so their text is missing from an export. */
export const partlyReadPages = (doc) => doc.pages.filter((p) => (p.unread_regions?.length ?? 0) > 0).length;

/** True when the reader is on but no confidence threshold is set, so every line is marked. */
export function everyLineFlagged(doc) {
  const { reader, calibration } = doc.engine;
  return reader !== 'none' && (!calibration || calibration.tau == null);
}
