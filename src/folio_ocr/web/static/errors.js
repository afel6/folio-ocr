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
