import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  answerWhere, blankState, elements, installPage, makeDoc, makeLine, stubElement, tick, waitForCall,
} from './fake-page.mjs';

const calls = installPage();
// The download link frees its address after a minute; do not let that hold the test run open.
const realSetTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, ms, ...args) => realSetTimeout(fn, ms, ...args).unref();
const base = '../../src/folio_ocr/web/static/';
const { getState, setState } = await import(`${base}store.js`);
const files = await import(`${base}files.js`);
const { msg } = await import(`${base}copy.js`);
const { openDoc } = await import(`${base}docs.js`);
const { confirm } = await import(`${base}lines.js`);

// Status of each reply, decided when the request is sent: [matcher, status] pairs.
let statuses = [];
const planned = (...pairs) => { statuses = pairs; };

beforeEach(() => {
  calls.length = 0;
  calls.created.length = 0;
  calls.appended.length = 0;
  elements.file = stubElement();
  elements['doc-title'] = stubElement();
  elements['docs-heading'] = stubElement();
  elements['start-title'] = stubElement();
  statuses = [];
  calls.statusFor = (c) => statuses.find(([match]) => match(c))?.[1] ?? 200;
  setState({ ...blankState, uploading: false, dialog: null });
});

const is = (method, re) => (c) => c.init.method === method && re.test(c.url);
const listGet = is('GET', /^\/api\/documents$/);
const post = is('POST', /^\/api\/documents$/);
const openGet = is('GET', /export\?format=json$/);
const del = is('DELETE', /^\/api\/documents\/[^/]+$/);
const reads = is('POST', /\/read$/);

/** Answer the next unanswered call that matches; waits for it to be sent. */
const reply = (match, body) => answerWhere(calls, match, body);

const sent = (match) => calls.filter(match).map((c) => c.url);
const pageOf = (index) => ({ index, status: 'unread', width: 10, height: 10, lines: [] });
const twoPageDoc = (id, filename) => makeDoc(id, [], { source: { filename }, pages: [pageOf(1), pageOf(2)] });

/** Upload `name` and answer the list and add calls; leaves the open call to the test. */
async function startUpload(name, id) {
  const done = files.upload({ name }, 'en');
  await reply(listGet, []);
  await reply(post, { document_id: id, page_count: 2 });
  await reply(listGet, [{ document_id: id, filename: name }]);
  return done;
}

/** Answer every read still waiting with a 400, which skips that page and goes on. */
async function drainReads() {
  for (let i = 0; i < 5 && calls.some((c) => !c.answered && reads(c)); i += 1) {
    await reply(reads, { error: 'x' });
  }
}

test('an upload whose open fails with a 500 still reads its pages one at a time', async () => {
  planned([reads, 400], [openGet, 500]);
  const done = startUpload('five.png', 'up500');
  await reply(openGet, { error: 'boom' });
  await done;
  assert.deepEqual(sent(reads), ['/api/documents/up500/pages/1/read']);
  await reply(reads, { error: 'x' });
  assert.deepEqual(sent(reads), ['/api/documents/up500/pages/1/read', '/api/documents/up500/pages/2/read']);
  await drainReads();
});

test('an upload whose open fails with a 404 sends no read', async () => {
  planned([openGet, 404]);
  const done = startUpload('four.png', 'up404');
  await reply(openGet, { error: 'gone' });
  await done;
  await tick();
  assert.deepEqual(sent(reads), []);
});

test('after an upload the status names the stored file and the page being read', async () => {
  planned([reads, 400]);
  const done = startUpload('raw name.png', 'upok');
  await reply(openGet, twoPageDoc('upok', 'clean name.png'));
  await done;
  assert.equal(getState().status, 'clean name.png added: 2 pages. Reading page 1 of 2…');
  assert.equal(getState().screen, 'doc');
  await drainReads();
});

test('the status never stays on "Adding" after an upload that succeeds', async () => {
  planned([reads, 400]);
  const done = files.upload({ name: 'a.png' }, 'en');
  await tick();
  assert.equal(getState().status, 'Adding a.png…');
  await reply(listGet, []);
  await reply(post, { document_id: 'upset', page_count: 2 });
  await reply(listGet, []);
  await reply(openGet, twoPageDoc('upset', 'a.png'));
  await done;
  assert.ok(!getState().status.startsWith('Adding'));
  await drainReads();
});

test('the status never stays on "Adding" after an upload that fails', async () => {
  planned([post, 409]);
  const done = files.upload({ name: 'a.png' }, 'en');
  await reply(listGet, []);
  await reply(post, { error: 'full' });
  await done;
  assert.equal(getState().status, '');
  assert.deepEqual(getState().alert, [msg.storeFull('a.png')]);
  assert.equal(getState().uploading, false);
});

function openTwo(first, second) {
  setState({
    screen: 'doc',
    docId: first,
    page: 1,
    lineId: 'l',
    list: [{ document_id: first }, { document_id: second }],
    docs: { [first]: makeDoc(first, [makeLine('l')]), ...(getState().docs) },
  });
}

test('a delete whose next document cannot be opened leaves the start screen with no document', async () => {
  openTwo('del1', 'del2');
  planned([del, 204], [openGet, 500]);
  const done = files.deleteDoc();
  await reply(del, undefined);
  await reply(openGet, { error: 'boom' });
  await done;
  assert.equal(getState().screen, 'start');
  assert.equal(getState().docId, null);
});

test('a delete finishing after the user moved to another document leaves that document open', async () => {
  openTwo('del3', 'del4');
  setState({
    list: [{ document_id: 'del3' }, { document_id: 'del4' }, { document_id: 'del5' }],
    docs: { ...getState().docs, del5: makeDoc('del5', [makeLine('l')]) },
  });
  planned([del, 204]);
  const done = files.deleteDoc();
  setState({ docId: 'del5' });
  await reply(del, undefined);
  await done;
  assert.equal(getState().screen, 'doc');
  assert.equal(getState().docId, 'del5');
  assert.equal(getState().status, 'del3.png deleted.');
});

test('export hands the file to the browser and says how many lines still need checking', async () => {
  const lines = [makeLine('p1-l1'), makeLine('p1-l2'), makeLine('p1-l3'), makeLine('p1-l4', { status: 'confirmed' })];
  setState({ screen: 'doc', docId: 'ex1', docs: { ex1: makeDoc('ex1', lines) } });
  const done = files.exportDoc('txt');
  await tick();
  calls[0].headers = new Headers({ 'X-Unresolved-Count': '7' });
  calls[0].answerBlob(new Blob(['text']));
  await done;
  assert.equal(calls[0].url, '/api/documents/ex1/export?format=txt');
  const [link] = calls.created;
  assert.equal(link.attrs.download, 'ex1.txt');
  assert.equal(link.clicked, 1);
  assert.deepEqual(calls.appended, [link]);
  assert.equal(getState().status, 'Exported ex1.txt. 3 lines still need checking.');
});

test('an export whose body is cut off shows the friendly alert and downloads nothing', async () => {
  setState({ screen: 'doc', docId: 'ex2', docs: { ex2: makeDoc('ex2', []) } });
  const done = files.exportDoc('txt');
  await tick();
  calls[0].failBlob(new Error('connection reset'));
  await done;
  assert.deepEqual(getState().alert, ['Couldn’t export ex2.png: folio didn’t answer. Try again.']);
  assert.equal(calls.created.length, 0);
});

/** A document with `toCheck` lines to check on page 1 and unread areas on `partly` more read pages. */
function exportable(id, toCheck, partly) {
  const lines = Array.from({ length: toCheck }, (_, i) => makeLine(`p1-l${i + 1}`));
  const extra = Array.from({ length: partly }, (_, i) => ({
    index: i + 2, status: 'read', width: 10, height: 10, lines: [], unread_regions: [[0, 0, 1, 1]],
  }));
  const doc = makeDoc(id, lines);
  return { ...doc, pages: [...doc.pages, ...extra] };
}

async function exportWithHeader(id, format, unresolved) {
  const done = files.exportDoc(format);
  await tick();
  calls[0].headers = new Headers({ 'X-Unresolved-Count': String(unresolved) });
  calls[0].answerBlob(new Blob(['x']));
  await done;
  return getState().status;
}

test('the export count says lines only for lines to check and names pages with unread areas apart', async () => {
  setState({ screen: 'doc', docId: 'ex3', docs: { ex3: exportable('ex3', 1, 1) } });
  assert.equal(await exportWithHeader('ex3', 'json', 2),
    'Exported ex3.json. 1 line still needs checking. Parts of 1 page weren’t read, so their text isn’t in the export.');
});

test('an export with no lines to check but unread areas says no lines are left and names those pages', async () => {
  setState({ screen: 'doc', docId: 'ex4', docs: { ex4: exportable('ex4', 0, 2) } });
  assert.equal(await exportWithHeader('ex4', 'docx', 2),
    'Exported ex4.docx. No lines left to check. Parts of 2 pages weren’t read, so their text isn’t in the export.');
});

test('the export count is the document as it was when the export was asked for', async () => {
  setState({ screen: 'doc', docId: 'ex5', docs: { ex5: exportable('ex5', 2, 0) } });
  const done = files.exportDoc('txt');
  await tick();
  setState({ docs: { ex5: exportable('ex5', 0, 0) } });
  calls[0].answerBlob(new Blob(['x']));
  await done;
  assert.equal(getState().status, 'Exported ex5.txt. 2 lines still need checking.');
});

test('the export count includes a save that was still running when the export was asked for', async () => {
  const lines = [makeLine('p1-l1'), makeLine('p1-l2')];
  setState({ screen: 'doc', docId: 'ex6', page: 1, lineId: 'p1-l1', docs: { ex6: makeDoc('ex6', lines) } });
  confirm();
  const done = files.exportDoc('txt');
  await reply(is('PUT', /\/lines\//), makeLine('p1-l1', { status: 'confirmed' }));
  const blobCall = await waitForCall(calls, is('GET', /export\?format=txt$/));
  blobCall.answerBlob(new Blob(['x']));
  await done;
  assert.equal(getState().status, 'Exported ex6.txt. 1 line still needs checking.');
});

test('a document still on its way when the open one is deleted does not replace the next one', async () => {
  openTwo('del6', 'del7');
  setState({ docs: { ...getState().docs, del7: makeDoc('del7', [makeLine('l')]) } });
  const slow = openDoc('del8');
  await tick();
  planned([del, 204]);
  const done = files.deleteDoc();
  await reply(del, undefined);
  await done;
  await reply((c) => c.url === '/api/documents/del8/export?format=json', makeDoc('del8', [makeLine('l')]));
  assert.equal(await slow, false);
  assert.equal(getState().screen, 'doc');
  assert.equal(getState().docId, 'del7');
});
