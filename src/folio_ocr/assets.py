"""Pinned downloadable assets (Tesseract language and page-orientation data, OnnxTR model
files): check them on disk and fetch them over https, saving a file only when its size and
sha256 match the pinned values."""

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import IO, NamedTuple
from urllib.parse import urlsplit

import httpx

from folio_ocr.errors import WeightsError

TESSDATA_DIR = "tessdata"  # Tesseract runs with --tessdata-dir <assets folder>/tessdata
_CHUNK = 1024 * 1024
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    size_bytes: int
    licence: str

    def __post_init__(self) -> None:
        parts = self.name.split("/")
        if (
            not self.name
            or self.name.startswith("/")
            or "\\" in self.name
            or any(p in ("", ".", "..") for p in parts)
        ):
            raise ValueError(f"asset name {self.name!r} is not a safe relative path")
        if len(self.sha256) != 64 or not set(self.sha256) <= _HEX:
            raise ValueError("sha256 must be 64 lowercase hex characters")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        url = urlsplit(self.url)
        if url.scheme != "https" or not url.hostname:
            raise ValueError("url must be an https URL with a host")


class AssetStatus(NamedTuple):
    asset: Asset
    ok: bool
    reason: str  # "ok", "missing", "wrong size", "sha256 mismatch" or "cannot read"


_TESSDATA_BEST = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/"
_BEST_COMMIT = "e12c65a915945e4c28e237a9b52bc4a8f39a0cec"
_TESSDATA_REPO = "https://raw.githubusercontent.com/tesseract-ocr/tessdata/"
# the osd file at this commit is the one bundled with Tesseract 5.4.0
_OSD_COMMIT = "13665561c0162cbd8d94d1f8cab50585849de998"
_ONNXTR = "https://github.com/felixdittrich92/OnnxTR/releases/download/"
ONNXTR_DIR = "onnxtr"  # OnnxTR model files live in <assets folder>/onnxtr
ONNXTR_DETECTOR = Asset(
    f"{ONNXTR_DIR}/rep_fast_base-1b89ebf9.onnx",
    f"{_ONNXTR}v0.0.1/rep_fast_base-1b89ebf9.onnx",
    "1b89ebf9b6a5e01eb21b1775efc54aab6dbf8c726dcce938e40ddcb91404b43a",
    42_342_927,
    "Apache-2.0",
)
ONNXTR_RECOGNISER = Asset(
    f"{ONNXTR_DIR}/crnn_vgg16_bn-743599aa.onnx",
    f"{_ONNXTR}v0.7.1/crnn_vgg16_bn-743599aa.onnx",
    "743599aa33cb4d254e14f3fe013f1c2a49eaff1329a7ec451844a0caa4f84bc5",
    63_213_845,
    "Apache-2.0",
)
TESSDATA_ARA = Asset(
    f"{TESSDATA_DIR}/ara.traineddata",
    f"{_TESSDATA_BEST}{_BEST_COMMIT}/ara.traineddata",
    "ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896",
    12_603_724,
    "Apache-2.0",
)
TESSDATA_ENG = Asset(
    f"{TESSDATA_DIR}/eng.traineddata",
    f"{_TESSDATA_BEST}{_BEST_COMMIT}/eng.traineddata",
    "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba",
    15_400_601,
    "Apache-2.0",
)
TESSERACT_OSD = Asset(  # page-orientation data, read by `tesseract --psm 0`
    f"{TESSDATA_DIR}/osd.traineddata",
    f"{_TESSDATA_REPO}{_OSD_COMMIT}/osd.traineddata",
    "9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff",
    10_562_727,
    "Apache-2.0",
)
# Language data the Tesseract reader needs; the orientation data is listed on its own.
TESSDATA: tuple[Asset, ...] = (TESSDATA_ARA, TESSDATA_ENG)
ONNXTR: tuple[Asset, ...] = (ONNXTR_DETECTOR, ONNXTR_RECOGNISER)
MANIFEST: tuple[Asset, ...] = (*TESSDATA, TESSERACT_OSD, *ONNXTR)


_REMEMBERED: ContextVar[dict[tuple[str, int, int, str], str] | None] = ContextVar(
    "folio_asset_verdicts", default=None
)


@contextmanager
def remembered(verdicts: dict) -> Iterator[None]:
    """Within the block, `verify` (and so `files_problem`) keeps each file's hash verdict in
    `verdicts` and reuses it while the file's size and modification time are unchanged."""
    token = _REMEMBERED.set(verdicts)
    try:
        yield
    finally:
        _REMEMBERED.reset(token)


def _reason(asset: Asset, path: Path) -> str:
    verdicts = _REMEMBERED.get()
    try:
        with open(path, "rb") as handle:
            info = os.fstat(handle.fileno())
            if info.st_size != asset.size_bytes:
                return "wrong size"
            key = (os.fspath(path), info.st_size, info.st_mtime_ns, asset.sha256)
            if verdicts is not None and key in verdicts:
                return verdicts[key]
            digest = hashlib.sha256()
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "cannot read"
    verdict = "ok" if digest.hexdigest() == asset.sha256 else "sha256 mismatch"
    if verdicts is not None:
        verdicts[key] = verdict
    return verdict


def files_problem(assets: Iterable[Asset], directory: Path) -> str | None:
    """None when every asset verifies, else the worst reason over them: "missing", then
    "cannot read" (an OS error such as permission denied), then "damaged" (wrong size or
    sha256)."""
    reasons = {status.reason for status in verify(assets, directory)}
    for reason in ("missing", "cannot read"):
        if reason in reasons:
            return reason
    return None if reasons <= {"ok"} else "damaged"


def verify(assets: Iterable[Asset], directory: Path) -> list[AssetStatus]:
    """Check each asset under `directory`; returns one status per asset, in input order."""
    result = []
    for asset in assets:
        reason = _reason(asset, directory / asset.name)
        result.append(AssetStatus(asset, reason == "ok", reason))
    return result


def _require_https(request: httpx.Request) -> None:
    if request.url.scheme != "https":
        raise WeightsError("refused a download that is not https")


def _say(report: Callable[[str], None] | None, message: str) -> None:
    if report is not None:
        report(message)


def _write_body(response: httpx.Response, handle: IO[bytes], asset: Asset) -> None:
    """Write the decoded body to `handle` (closed on return) and check size and sha256."""
    digest, count = hashlib.sha256(), 0
    try:
        for chunk in response.iter_bytes():
            handle.write(chunk)
            digest.update(chunk)
            count += len(chunk)
            if count > asset.size_bytes:
                break
    finally:
        handle.close()
    if count != asset.size_bytes or digest.hexdigest() != asset.sha256:
        raise WeightsError(
            f"{asset.name} does not match its pinned size and sha256; nothing was saved"
        )


def _download(client: httpx.Client, asset: Asset, target: Path) -> None:
    temp: Path | None = None
    try:
        with client.stream("GET", asset.url) as response:
            if response.status_code != 200:
                raise WeightsError(
                    f"cannot download {asset.name}: the server answered HTTP {response.status_code}"
                )
            handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 (closed in _write_body)
                dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False
            )
            temp = Path(handle.name)
            _write_body(response, handle, asset)
        os.replace(temp, target)
    except httpx.TimeoutException as exc:
        raise WeightsError(f"cannot download {asset.name}: the connection timed out") from exc
    except httpx.HTTPError as exc:
        raise WeightsError(f"cannot download {asset.name}: network error") from exc
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def fetch(
    assets: Iterable[Asset],
    directory: Path,
    *,
    report: Callable[[str], None] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Download every asset that is missing or fails verification; stops at the first error."""
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for status in verify(assets, directory):
        if status.ok:
            _say(report, f"{status.asset.name}: already verified")
        else:
            pending.append(status.asset)
    with httpx.Client(
        transport=transport,
        follow_redirects=True,
        timeout=httpx.Timeout(30.0),
        event_hooks={"request": [_require_https]},
    ) as client:
        for asset in pending:
            _say(report, f"{asset.name}: downloading {asset.size_bytes / 1e6:.1f} MB")
            target = directory / asset.name
            target.parent.mkdir(parents=True, exist_ok=True)
            _download(client, asset, target)
            _say(report, f"{asset.name}: verified")
