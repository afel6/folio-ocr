/** A failed call to folio: no answer (`network`), an error status (`http`) or an unreadable body (`badBody`). */
export class ApiError extends Error {
  constructor(kind, status = null, serverLine = null) {
    super(`${kind}${status === null ? '' : ` ${status}`}`);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.serverLine = serverLine;
  }
}

/** The exact line folio sends with a 503 when no reader is free in time. */
export const READERS_BUSY_LINE = 'folio is busy reading other files; try again shortly';

/** True for folio's reply that every reader is busy: a passing state, worth trying again. */
export const isReadersBusy = (err) => err instanceof ApiError && err.kind === 'http'
  && err.status === 503 && err.serverLine === READERS_BUSY_LINE;
