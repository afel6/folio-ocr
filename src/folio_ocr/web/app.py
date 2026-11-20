"""The HTTP API of the local web service and the pages it serves."""

import dataclasses
import io
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, StrictStr
from starlette.exceptions import HTTPException
from starlette.routing import Match, Mount
from starlette.staticfiles import StaticFiles

from folio_ocr import assets, export, geometry, model_health, ocr_setup, pipeline
from folio_ocr.config import Settings
from folio_ocr.errors import InputError, ModelServerError, WeightsError
from folio_ocr.model_reader import ModelReader
from folio_ocr.web.guard import Guard, allowed_hosts, stderr_line
from folio_ocr.web.store import ApiError, Store
from folio_ocr.web.upload import new_document, parse_form, read_body

STATIC_DIR = Path(__file__).parent / "static"
BAD_BODY = 'the request body is not valid; send JSON like {"final_text": "..."}'
BAD_TEXT = "final_text holds a character that is not valid text"
BAD_FORMAT = "format must be txt, docx or json"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
HTTP_MESSAGES = {404: "not found", 405: "method not allowed"}


class _PagesMount(Mount):
    """The static pages: never answers a path under /api, so a wrong method on an API path gets
    the API's own 405 and an unknown API path gets a 404."""

    def matches(self, scope):
        path = scope["path"]
        if path == "/api" or path.startswith("/api/"):
            return Match.NONE, {}
        return super().matches(scope)


class LineEdit(BaseModel):
    final_text: StrictStr


def _reply(status: int, message: str, headers=None) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers=headers)


def _allowed_methods(request: Request) -> str:
    """Every method any route accepts for the path of the request."""
    methods: set[str] = set()
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.PARTIAL:
            methods |= getattr(route, "methods", None) or set()
    return ", ".join(sorted(methods))


def _add_error_handlers(app: FastAPI) -> None:
    def fixed(exc_type: type[Exception], status: int) -> None:
        app.add_exception_handler(exc_type, lambda request, exc: _reply(status, str(exc)))

    app.add_exception_handler(ApiError, lambda request, exc: _reply(exc.status, str(exc)))
    fixed(InputError, 400)
    fixed(WeightsError, 503)  # includes SetupNeededError
    fixed(ModelServerError, 502)
    app.add_exception_handler(RequestValidationError, lambda request, exc: _reply(400, BAD_BODY))

    def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        message = HTTP_MESSAGES.get(exc.status_code, str(exc.detail))
        headers = getattr(exc, "headers", None)
        if exc.status_code == 405 and (allowed := _allowed_methods(request)):
            headers = {"Allow": allowed}  # a static file keeps its own "GET, HEAD"
        return _reply(exc.status_code, message, headers)

    app.add_exception_handler(HTTPException, http_error)


def _self_test(model: ModelReader, log: Callable[[str], None]) -> None:
    try:
        model.fit()
        model.server()  # the server's identity, so health knows whether it is calibrated
    except Exception as exc:  # noqa: BLE001 - reported on one line; the service keeps running
        log(f"error: the model server self-test could not run ({type(exc).__name__})")


def _lifespan(model: ModelReader | None, log: Callable[[str], None]):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model is not None:
            threading.Thread(target=_self_test, args=(model, log), daemon=True).start()
        try:
            yield
        finally:
            app.state.store.clear()

    return lifespan


def _summary(doc) -> dict:
    return {
        "document_id": doc.id,
        "filename": doc.source.filename,
        "page_count": doc.source.page_count,
        "languages": doc.languages,
        "created_at": doc.created_at.isoformat(),
        "has_edits": doc.has_edits,
        "unresolved_count": export.unresolved_count(doc),
        "pages": [{"index": p.index, "status": p.status} for p in doc.pages],
    }


def _add_status_routes(app: FastAPI, settings: Settings, tesseract, model) -> None:
    verdicts: dict = {}

    @app.get("/api/health")
    def health() -> dict:
        with assets.remembered(verdicts):
            tesseract_status = ocr_setup.tesseract_status(settings.weights_dir, tesseract)
            weights_status = ocr_setup.onnxtr_status(settings.weights_dir, tesseract)
        return {
            "status": "ok",
            "model": dataclasses.asdict(model_health.model_health(settings, model)),
            "tesseract": tesseract_status,
            "weights": weights_status,
        }

    @app.get("/api/documents")
    def documents() -> list:
        return [_summary(d) for d in app.state.store.documents()]


def _add_upload_routes(app: FastAPI) -> None:
    @app.post("/api/documents")
    async def upload(request: Request) -> dict:
        body = await read_body(request.stream(), request.headers.get("content-length"))
        form = await run_in_threadpool(parse_form, request.headers.get("content-type", ""), body)
        doc = await run_in_threadpool(new_document, form)
        app.state.store.add(form.data, doc)
        return {"document_id": doc.id, "page_count": doc.source.page_count}

    @app.delete("/api/documents/{doc_id}")
    def delete(doc_id: str) -> Response:
        app.state.store.delete(doc_id)
        return Response(status_code=204)


def _add_page_routes(app: FastAPI, settings: Settings, tesseract, model, read_page, page_image):
    store: Store = app.state.store

    @app.get("/api/documents/{doc_id}/pages/{n}/image")
    def image(doc_id: str, n: str) -> Response:
        entry, page = store.page(doc_id, n)
        straight, _ = geometry.straighten(page_image(entry.data, page.index), page.rotation_applied)
        png = io.BytesIO()
        straight.save(png, "PNG")
        return Response(png.getvalue(), media_type="image/png")

    @app.post("/api/documents/{doc_id}/pages/{n}/read")
    def read(doc_id: str, n: str) -> JSONResponse:
        entry, no = store.begin_read(doc_id, n)
        try:
            page, engine = read_page(
                entry.data,
                no,
                languages=entry.doc.languages,
                weights_dir=settings.weights_dir,
                tesseract_path=tesseract,
                line_timeout_s=settings.line_timeout_s,
                model=model,
            )
        except BaseException:
            store.abort_read()
            raise
        return JSONResponse(dataclasses.asdict(store.finish_read(page, engine)))

    @app.put("/api/documents/{doc_id}/lines/{line_id}")
    def edit(doc_id: str, line_id: str, body: LineEdit, request: Request) -> JSONResponse:
        media_type = request.headers.get("content-type")
        if (
            media_type is not None
            and media_type.split(";")[0].strip().lower() != "application/json"
        ):
            raise ApiError(400, BAD_BODY)
        try:
            body.final_text.encode("utf-8")
        except UnicodeEncodeError:
            raise ApiError(400, BAD_TEXT) from None
        return JSONResponse(dataclasses.asdict(store.edit_line(doc_id, line_id, body.final_text)))


def _attachment(name: str) -> str:
    ascii_name = "".join(ch if ch.isascii() else "_" for ch in name)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name, safe='')}"


def _export_body(fmt: str, doc) -> tuple[Callable[[], bytes], str]:
    if fmt == "txt":
        return lambda: export.to_txt(doc).encode("utf-8"), "text/plain; charset=utf-8"
    if fmt == "docx":
        return lambda: export.to_docx(doc), DOCX
    return lambda: export.to_json(doc, indent=2).encode("utf-8"), "application/json"


def _add_export_route(app: FastAPI) -> None:
    @app.get("/api/documents/{doc_id}/export")
    async def export_document(doc_id: str, request: Request) -> Response:
        fmt = request.query_params.get("format")
        if fmt not in ("txt", "docx", "json"):
            raise ApiError(400, BAD_FORMAT)
        doc = app.state.store.get(doc_id).doc
        make, media_type = _export_body(fmt, doc)
        headers = {
            "Content-Disposition": _attachment(f"{Path(doc.source.filename).stem}.{fmt}"),
            "X-Unresolved-Count": str(export.unresolved_count(doc)),
            "X-Unread-Pages": str(sum(p.status != "read" for p in doc.pages)),
        }
        return Response(await run_in_threadpool(make), media_type=media_type, headers=headers)


def create_app(
    settings: Settings,
    *,
    port: int,
    tesseract: Path | None,
    model: ModelReader | None,
    read_page=pipeline.read_page,
    page_image=pipeline.page_image,
    static_dir: Path = STATIC_DIR,
    log: Callable[[str], None] = stderr_line,
) -> FastAPI:
    """The web service: the JSON API under /api and the static pages at /."""
    app = FastAPI(
        lifespan=_lifespan(model, log),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        strict_content_type=False,
        redirect_slashes=False,
    )
    app.state.store = Store()
    app.add_middleware(Guard, hosts=allowed_hosts(settings.host, port), log=log)
    _add_error_handlers(app)
    _add_status_routes(app, settings, tesseract, model)
    _add_upload_routes(app)
    _add_page_routes(app, settings, tesseract, model, read_page, page_image)
    _add_export_route(app)
    app.router.routes.append(_PagesMount("/", app=StaticFiles(directory=static_dir, html=True)))
    return app
