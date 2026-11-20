// The one copy of the screen state. Every change makes a new state object and redraws.

let state = {
  health: null,
  healthError: null,
  list: [],
  docs: {},
  screen: 'start',
  docId: null,
  page: 1,
  lineId: null,
  draft: null,
  queue: [],
  inFlight: null,
  paused: false,
  readCounts: {},
  imageFailed: {},
  status: '',
  alert: null,
  dialog: null,
  uploading: false,
};

const listeners = [];

export const getState = () => state;

/** Merge a patch (or a function of the state returning one) into a new state, then redraw. */
export function setState(patch) {
  const change = typeof patch === 'function' ? patch(state) : patch;
  state = { ...state, ...change };
  for (const listener of listeners) listener(state);
}

export function subscribe(listener) {
  listeners.push(listener);
}
