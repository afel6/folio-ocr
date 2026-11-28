"""Page images and uploads that wait for a reader slot hold no worker thread, so other routes
keep answering; waits (5 s per slot) and slow upload bodies (30 s idle, 600 s in all) are
bounded, and a page image of a deleted document is 404."""

import asyncio
import threading
import time

import anyio
import anyio.to_thread
import httpx
import pytest
from PIL import Image
from sample_files import save
from starlette.testclient import TestClient

from folio_ocr import pdf_client
from folio_ocr.config import Settings
from folio_ocr.web import app as web_app
from folio_ocr.web import upload
from folio_ocr.web.app import create_app
from folio_ocr.web.store import DOCUMENT_GONE

PNG = save(Image.new("L", (10, 10)), "PNG")
BUSY = {"error": "folio is busy reading other files; try again shortly"}
SLOW = {"error": "the upload took too long; try again"}
READER_WAIT_S = 5  # a slot a request needs must come free within 5 s
IDLE_LIMIT_S = 30  # an upload body may pause at most 30 s between bytes
TOTAL_LIMIT_S = 600  # a whole upload body must arrive within 600 s
HANG_S = 60  # ends a test that hangs; no wait here comes near it
GATE_HANG_S = 120  # a gate no test opens still opens, after HANG_S so the hang fails first
QUEUE_WAIT_S = 120  # far above the time the queue tests take; the 5 s wait has tests of its own
TICK_S = 0.01  # a real pause after each clock move, so limits the clock has passed fire first


class Gate:
    """A stand-in reader that blocks until released and records how many run at once."""

    def __init__(self):
        self.open = threading.Event()
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def enter(self):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            self.open.wait(GATE_HANG_S)
        finally:
            with self.lock:
                self.running -= 1

    def page_image(self, data, no):
        self.enter()
        return Image.new("L", (40, 20))

    def page_count(self, data, kind):
        self.enter()
        return 1

    def wait_running(self, count):
        wait_until(lambda: self.running >= count)


def wait_until(check):
    end = time.monotonic() + HANG_S
    while not check():
        assert time.monotonic() < end
        time.sleep(0.02)


@pytest.fixture
def gate():
    g = Gate()
    yield g
    g.open.set()


def make_app(tmp_path, **readers):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>home</p>")
    return create_app(
        Settings(weights_dir=tmp_path), port=8000, tesseract=None, model=None,
        static_dir=static, log=lambda line: None, **readers,
    )  # fmt: skip


@pytest.fixture
def client(tmp_path, gate):
    app = make_app(tmp_path, read_page=None, page_image=gate.page_image)
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        yield c


def post_png(c):
    return c.post(
        "/api/documents", files={"file": ("a.png", PNG, "application/x")}, data={"languages": "en"}
    )


def image_url(doc_id):
    return f"/api/documents/{doc_id}/pages/1/image"


def pool_size(client):
    limiter = anyio.to_thread.current_default_thread_limiter
    return int(client.portal.call(lambda: limiter().total_tokens))


def waiting_for(client, slot):
    """How many requests wait for `slot` (web_app._READERS or web_app._UPLOADS)."""
    return client.portal.call(lambda: web_app._limiter(slot).statistics().tasks_waiting)


def in_thread(fn, *args):
    out = []
    t = threading.Thread(target=lambda: out.append(fn(*args)))
    t.start()
    return t, out


def fire(client, count, send):
    """Send `count` requests at once; their threads and the status codes received so far."""
    codes = []

    def run():
        codes.append(send(client).status_code)

    threads = [threading.Thread(target=run) for _ in range(count)]
    for t in threads:
        t.start()
    return threads, codes


def health_codes(client):
    t, out = in_thread(client.get, "/api/health")
    t.join(timeout=HANG_S)
    return [r.status_code for r in out]


def finish(gate, threads):
    gate.open.set()
    for t in threads:
        t.join(timeout=HANG_S)


def make_document(client, monkeypatch):
    monkeypatch.setattr(upload, "_page_count", lambda data, kind: 2)
    return post_png(client).json()["document_id"]


def assert_gone(res):
    assert res.status_code == 404
    assert res.headers["content-type"] == "application/json"
    assert res.json() == {"error": DOCUMENT_GONE}


def test_waiting_page_images_leave_other_routes_answering(client, gate, monkeypatch):
    monkeypatch.setattr(web_app, "_READER_WAIT_S", QUEUE_WAIT_S)
    doc_id = make_document(client, monkeypatch)
    threads, codes = fire(client, pool_size(client) + 2, lambda c: c.get(image_url(doc_id)))
    queued = len(threads) - pdf_client.MAX_WORKERS
    try:
        gate.wait_running(pdf_client.MAX_WORKERS)
        wait_until(lambda: waiting_for(client, web_app._READERS) == queued)
        assert health_codes(client) == [200]
        assert codes == []
    finally:
        finish(gate, threads)
    assert codes == [200] * len(threads)
    assert gate.peak <= pdf_client.MAX_WORKERS


def test_waiting_uploads_leave_other_routes_answering(client, gate, monkeypatch):
    monkeypatch.setattr(web_app, "_READER_WAIT_S", QUEUE_WAIT_S)
    monkeypatch.setattr(upload, "_page_count", gate.page_count)
    threads, codes = fire(client, pool_size(client) + 2, post_png)
    queued = len(threads) - pdf_client.MAX_WORKERS
    try:
        gate.wait_running(pdf_client.MAX_WORKERS)
        wait_until(lambda: waiting_for(client, web_app._UPLOADS) == queued)
        assert health_codes(client) == [200]
        assert codes == []
    finally:
        finish(gate, threads)
    assert codes == [200] * len(threads)
    assert gate.peak <= pdf_client.MAX_WORKERS


def test_an_unknown_document_answers_404_while_readers_are_busy(client, gate, monkeypatch):
    doc_id = make_document(client, monkeypatch)
    threads, _ = fire(client, pdf_client.MAX_WORKERS + 1, lambda c: c.get(image_url(doc_id)))
    try:
        gate.wait_running(pdf_client.MAX_WORKERS)
        t, missing = in_thread(client.get, image_url("0" * 32))
        threads.append(t)
        t.join(timeout=HANG_S)
        assert [r.status_code for r in missing] == [404]
    finally:
        finish(gate, threads)


def test_a_waiting_page_image_gets_a_busy_line_before_a_reader_is_released(
    client, gate, monkeypatch
):
    monkeypatch.setattr(web_app, "_READER_WAIT_S", 0.1)
    doc_id = make_document(client, monkeypatch)
    threads, codes = fire(client, pdf_client.MAX_WORKERS, lambda c: c.get(image_url(doc_id)))
    try:
        gate.wait_running(pdf_client.MAX_WORKERS)
        t, waiting = in_thread(client.get, image_url(doc_id))
        threads.append(t)
        t.join(timeout=HANG_S)
        assert [(r.status_code, r.json()) for r in waiting] == [(503, BUSY)]
        assert health_codes(client) == [200]
        assert codes == []
    finally:
        finish(gate, threads)
    assert client.get(image_url(doc_id)).status_code == 200


def test_upload_gives_a_busy_line_when_the_native_reader_slots_are_held(client, monkeypatch):
    monkeypatch.setattr(web_app, "_READER_WAIT_S", 0.1)
    held, threads = 0, []
    try:
        for _ in range(pdf_client.MAX_WORKERS):
            assert pdf_client._SLOTS.acquire(timeout=HANG_S)
            held += 1
        t, waiting = in_thread(post_png, client)
        threads.append(t)
        t.join(timeout=HANG_S)
        assert [(r.status_code, r.json()) for r in waiting] == [(503, BUSY)]
        assert client.get("/api/documents").json() == []
    finally:
        for _ in range(held):
            pdf_client._SLOTS.release()
        for t in threads:
            t.join(timeout=HANG_S)


def test_receiving_upload_bodies_does_not_occupy_page_image_slots(client, gate, monkeypatch):
    doc_id = make_document(client, monkeypatch)
    gate.open.set()
    receiving, release = threading.Event(), threading.Event()
    starts = 0

    async def slow_body(*args):
        nonlocal starts
        starts += 1
        if starts == pdf_client.MAX_WORKERS:
            receiving.set()
        while not release.is_set():
            await anyio.sleep(0.02)
        return b""

    monkeypatch.setattr(web_app, "read_body", slow_body)
    monkeypatch.setattr(web_app, "_READER_WAIT_S", 0.1)
    uploads, _ = fire(client, pdf_client.MAX_WORKERS, lambda c: c.post("/api/documents"))
    try:
        assert receiving.wait(HANG_S)
        assert client.get(image_url(doc_id)).status_code == 200
    finally:
        release.set()
        finish(gate, uploads)


class NoFreeSlot:
    """Stand-in for the native reader slots: none comes free; records each wait asked for."""

    def __init__(self):
        self.waits = []

    def acquire(self, timeout=None):
        self.waits.append(timeout)
        return False


NATIVE_ROUTES = {
    "upload": lambda c, doc_id: post_png(c),
    "page image": lambda c, doc_id: c.get(image_url(doc_id)),
    "page read": lambda c, doc_id: c.post(f"/api/documents/{doc_id}/pages/1/read"),
}


@pytest.mark.parametrize("route", NATIVE_ROUTES)
def test_each_reader_route_waits_5_s_for_a_native_reader_slot(tmp_path, monkeypatch, route):
    send = NATIVE_ROUTES[route]
    slots = NoFreeSlot()
    with TestClient(make_app(tmp_path), base_url="http://127.0.0.1:8000") as c:
        with monkeypatch.context() as m:
            m.setattr(upload, "_page_count", lambda data, kind: 1)
            doc_id = post_png(c).json()["document_id"]
        monkeypatch.setattr(pdf_client, "_SLOTS", slots)
        replies = [send(c, doc_id), send(c, doc_id)]  # a failed read leaves no read running
        docs = c.get("/api/documents").json()
    assert [(r.status_code, r.json()) for r in replies] == [(503, BUSY), (503, BUSY)]
    assert slots.waits == [READER_WAIT_S, READER_WAIT_S]
    assert [(d["document_id"], [p["status"] for p in d["pages"]]) for d in docs] == [
        (doc_id, ["unread"])
    ]


def test_the_limits_are_5_s_per_wait_30_s_idle_and_600_s_per_upload_body():
    limits = (web_app._READER_WAIT_S, web_app._UPLOAD_IDLE_S, web_app._UPLOAD_TOTAL_S)
    assert limits == (READER_WAIT_S, IDLE_LIMIT_S, TOTAL_LIMIT_S)


def test_a_page_image_of_a_document_deleted_while_it_waits_for_a_reader_is_404(
    client, gate, monkeypatch
):
    monkeypatch.setattr(web_app, "_READER_WAIT_S", QUEUE_WAIT_S)
    doc_id = make_document(client, monkeypatch)
    other = make_document(client, monkeypatch)
    threads, codes = fire(client, pdf_client.MAX_WORKERS, lambda c: c.get(image_url(other)))
    try:
        gate.wait_running(pdf_client.MAX_WORKERS)
        t, got = in_thread(client.get, image_url(doc_id))
        threads.append(t)
        wait_until(lambda: waiting_for(client, web_app._READERS) == 1)
        assert got == []
        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    finally:
        finish(gate, threads)
    assert codes == [200] * pdf_client.MAX_WORKERS
    assert_gone(got[0])


def test_a_page_image_of_a_document_deleted_while_it_renders_is_404(client, gate, monkeypatch):
    doc_id = make_document(client, monkeypatch)
    t, got = in_thread(client.get, image_url(doc_id))
    try:
        gate.wait_running(1)
        assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    finally:
        finish(gate, [t])
    assert_gone(got[0])


class ClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock a test moves forward."""

    offset = 0.0

    def time(self):
        return super().time() + self.offset


def run_on_clock_loop(main, *args):
    return anyio.run(main, *args, backend_options={"loop_factory": ClockLoop})


def quick_app(tmp_path, monkeypatch):
    monkeypatch.setattr(upload, "_page_count", lambda data, kind: 1)
    return make_app(tmp_path)


def asgi_client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000")


def multipart_body():
    req = httpx.Request(
        "POST", "http://127.0.0.1:8000/api/documents",
        files={"file": ("a.png", PNG, "application/x")}, data={"languages": "en"},
    )  # fmt: skip
    return req.read(), req.headers["content-type"]


def split(body, count):
    return [body[len(body) * i // count : len(body) * (i + 1) // count] for i in range(count)]


async def post_paced(app, gaps):
    """Post an upload in len(gaps) pieces. Before piece i the clock moves gaps[i] s, and every
    limit the clock has passed fires before the piece is sent."""
    loop = asyncio.get_running_loop()
    body, content_type = multipart_body()

    async def chunks():
        for gap, piece in zip(gaps, split(body, len(gaps)), strict=True):
            loop.offset += gap
            await asyncio.sleep(TICK_S)
            yield piece

    async with asgi_client(app) as c:
        return await c.post(
            "/api/documents", content=chunks(), headers={"content-type": content_type}
        )


async def paced_upload(app, gaps):
    res = await post_paced(app, gaps)
    return res, app.state.store.documents()


INSIDE_BOTH = [10] + [IDLE_LIMIT_S - 1] * 20  # pauses of 29 s, 590 s in all
STALL = [1, IDLE_LIMIT_S + 1, 1]  # one pause of 31 s


def test_a_slow_steady_upload_inside_both_limits_is_accepted(tmp_path, monkeypatch):
    app = quick_app(tmp_path, monkeypatch)
    res, docs = run_on_clock_loop(paced_upload, app, INSIDE_BOTH)
    assert res.status_code == 200
    assert [d.id for d in docs] == [res.json()["document_id"]]


def test_an_upload_that_stalls_past_the_idle_limit_gets_408(tmp_path, monkeypatch):
    app = quick_app(tmp_path, monkeypatch)
    res, docs = run_on_clock_loop(paced_upload, app, STALL)
    assert (res.status_code, res.json()) == (408, SLOW)
    assert docs == []


def test_a_trickling_upload_past_the_total_ceiling_gets_408(tmp_path, monkeypatch):
    app = quick_app(tmp_path, monkeypatch)
    past_total = [*INSIDE_BOTH, 20]  # the clock reads 610 s at the last piece
    res, docs = run_on_clock_loop(paced_upload, app, past_total)
    assert (res.status_code, res.json()) == (408, SLOW)
    assert docs == []


def test_upload_slots_are_freed_after_408(tmp_path, monkeypatch):
    app = quick_app(tmp_path, monkeypatch)

    async def main():
        codes = [(await post_paced(app, STALL)).status_code for _ in range(pdf_client.MAX_WORKERS)]
        held = web_app._limiter(web_app._UPLOADS).borrowed_tokens
        codes.append((await post_paced(app, [0])).status_code)
        return codes, held

    codes, held = run_on_clock_loop(main)
    assert codes == [408] * pdf_client.MAX_WORKERS + [200]
    assert held == 0


def hold(limiter):
    """Take every token of `limiter` for holders that are not requests (0, 1, ...)."""
    for holder in range(pdf_client.MAX_WORKERS):
        limiter.acquire_on_behalf_of_nowait(holder)


def release_all(limiter):
    for holder in range(pdf_client.MAX_WORKERS):
        limiter.release_on_behalf_of(holder)


def one_waiting(limiter):
    return lambda: limiter.statistics().tasks_waiting == 1


async def wait_on_loop(check):
    end = time.monotonic() + HANG_S
    while not check():
        assert time.monotonic() < end
        await asyncio.sleep(TICK_S)


async def post_counted(c, pulled):
    """An upload whose body records when the server first reads it."""
    body, content_type = multipart_body()

    async def chunks():
        pulled.append(True)
        yield body

    return await c.post("/api/documents", content=chunks(), headers={"content-type": content_type})


SLOT_WAITS = {  # slot held, request, whether the upload body was read
    "page image for a reader": (web_app._READERS, lambda c, d, p: c.get(image_url(d)), []),
    "upload for an upload slot": (web_app._UPLOADS, lambda c, d, p: post_counted(c, p), []),
    "upload for a reader": (web_app._READERS, lambda c, d, p: post_counted(c, p), [True]),
}


@pytest.mark.parametrize("case", SLOT_WAITS)
def test_a_request_waits_5_s_for_a_slot_then_gets_the_busy_line(tmp_path, monkeypatch, case):
    slot, send, body_read = SLOT_WAITS[case]
    app = quick_app(tmp_path, monkeypatch)

    async def main():
        loop = asyncio.get_running_loop()
        limiter, pulled, seen = web_app._limiter(slot), [], []
        async with asgi_client(app) as c:
            doc_id = (await post_png(c)).json()["document_id"]
            hold(limiter)
            waiting = asyncio.create_task(send(c, doc_id, pulled))
            await wait_on_loop(one_waiting(limiter))
            for move in (READER_WAIT_S - 1, 2):  # the clock reads 4 s, then 6 s
                loop.offset += move
                await asyncio.sleep(TICK_S)
                seen.append(limiter.statistics().tasks_waiting)
            release_all(limiter)  # a request still waiting now gets its slot and ends
            res = await waiting
        return seen, res, pulled, doc_id, [d.id for d in app.state.store.documents()]

    seen, res, pulled, doc_id, stored = run_on_clock_loop(main)
    assert seen == [1, 0]
    assert (res.status_code, res.json()) == (503, BUSY)
    assert pulled == body_read
    assert stored == [doc_id]


def test_an_upload_waits_up_to_5_s_for_each_slot_on_its_own(tmp_path, monkeypatch):
    app = quick_app(tmp_path, monkeypatch)

    async def main():
        loop = asyncio.get_running_loop()
        slots = [web_app._limiter(web_app._UPLOADS), web_app._limiter(web_app._READERS)]
        for limiter in slots:
            hold(limiter)
        async with asgi_client(app) as c:
            waiting = asyncio.create_task(post_png(c))
            for limiter in slots:  # 4 s on each, 8 s in all
                await wait_on_loop(one_waiting(limiter))
                loop.offset += READER_WAIT_S - 1
                await asyncio.sleep(TICK_S)
                limiter.release_on_behalf_of(0)
            res = await waiting
        return res, [d.id for d in app.state.store.documents()]

    res, stored = run_on_clock_loop(main)
    assert res.status_code == 200
    assert stored == [res.json()["document_id"]]
