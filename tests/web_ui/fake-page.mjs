// Test support: the browser globals the orchestration modules touch, and a fetch whose
// replies the test settles one by one.
export const tick = () => new Promise((resolve) => setImmediate(resolve));

export function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

/** Page elements by id, as `document.getElementById` returns them; clear between tests. */
export const elements = {};

/** An element stub that counts focus calls and keeps the attributes set on it. */
export function stubElement(extra = {}) {
  const el = {
    hidden: false, value: '', focused: 0, attrs: {}, clicked: 0, removed: 0,
    focus() { this.focused += 1; },
    setSelectionRange() {},
    setAttribute(name, value) { this.attrs[name] = value; },
    removeAttribute(name) { delete this.attrs[name]; },
    append() {},
    click() { this.clicked += 1; },
    remove() { this.removed += 1; },
    ...extra,
  };
  return el;
}

/**
 * Install the globals; returns `calls`, each { url, init, status, headers, answered,
 * answer(body), failBody(err), answerBlob(blob), failBlob(err) }. `calls.created` lists the elements the
 * page created, `calls.appended` the ones it added to the body. A reply's status is read as
 * soon as the request is sent, so set `calls.statusFor({ url, init })` before it (default 200).
 * With `calls.holdHeaders = true`, fetch() itself waits until the test calls `call.sendHeaders()`.
 */
export function installPage() {
  const calls = [];
  const fake = (url, init) => {
    const d = deferred();
    const b = deferred();
    const headersSent = deferred();
    const call = {
      url, init, status: calls.statusFor({ url, init }), headers: new Headers(), answered: false,
      answer: (body) => { call.answered = true; d.resolve(body); },
      failBody: (err) => { call.answered = true; d.reject(err); },
      answerBlob: b.resolve,
      failBlob: b.reject,
      sendHeaders: () => headersSent.resolve(),
    };
    calls.push(call);
    const res = {
      get ok() { return call.status < 400; },
      get status() { return call.status; },
      get headers() { return call.headers; },
      json: () => d.promise,
      blob: () => b.promise,
    };
    return calls.holdHeaders ? headersSent.promise.then(() => res) : Promise.resolve(res);
  };
  globalThis.window = {
    fetch: fake,
    location: { hash: '', pathname: '/', search: '' },
    history: { replaceState() {} },
  };
  calls.statusFor = () => 200;
  calls.holdHeaders = false;
  calls.created = [];
  calls.appended = [];
  globalThis.document = {
    getElementById: (id) => elements[id] ?? null,
    createElement: () => {
      const el = stubElement();
      calls.created.push(el);
      return el;
    },
    createTextNode: (text) => ({ text }),
    body: { append: (el) => calls.appended.push(el) },
  };
  URL.createObjectURL = () => 'blob:folio-test';
  URL.revokeObjectURL = () => {};
  return calls;
}

export const makeLine = (id, extra = {}) => ({
  id, status: 'needs_review', flags: [], text: 't', final_text: 't', direction: 'ltr', script: 'latin', ...extra,
});

export const makeDoc = (id, lines, extra = {}) => ({
  id,
  source: { filename: `${id}.png` },
  engine: { reader: 'onnxtr', calibration: { tau: 0.5 } },
  pages: [{ index: 1, status: 'read', width: 10, height: 10, lines }],
  ...extra,
});

export const blankState = {
  health: null, healthError: null, list: [], docs: {}, screen: 'start', docId: null, page: 1,
  lineId: null, draft: null, readCounts: {}, status: '', alert: null,
};

/** The first unanswered call `match` accepts, waiting a few turns for it to be sent. */
export async function waitForCall(calls, match) {
  for (let i = 0; i < 30; i += 1) {
    const call = calls.find((c) => !c.answered && match(c));
    if (call) return call;
    await tick();
  }
  throw new Error('expected call was not sent');
}

/** Answer the next unanswered call that `match` accepts, waiting for it to be sent. */
export async function answerWhere(calls, match, body) {
  const call = await waitForCall(calls, match);
  call.answer(body);
  await tick();
  return call;
}
