"""Settings read from environment variables.

Every value is checked once, when the settings are built. An error names the variable and
never repeats its value, because a model URL or key can hold a password.
"""

from __future__ import annotations

import ipaddress
import math
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from folio_ocr.calibration import version_major_minor
from folio_ocr.errors import ConfigError

_HOST_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?")
_WHOLE_NUMBER = re.compile(r"[0-9]+")
_SECONDS = re.compile(r"[0-9]+(\.[0-9]+)?")
_DIGEST = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
_SERVER_KIND = re.compile(r"[a-z0-9][a-z0-9._-]*")


@dataclass(frozen=True)
class DeclaredIdentity:
    """The model build and server software an administrator declares for a model server that
    does not report them itself."""

    kind: str  # the server software, e.g. "vllm"
    version: str  # major.minor when the declared version looks like one, else as declared
    digest: str  # the model build: 40 or 64 lower-case hex digits


@dataclass(frozen=True)
class Settings:
    """What the tool needs to run: where its files are, which model server to ask, and limits."""

    weights_dir: Path
    model_url: str | None = field(default=None, repr=False)  # may hold user:password
    model_name: str | None = None
    model_api_key: str | None = field(default=None, repr=False)
    tesseract_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    line_timeout_s: float = 60.0
    page_deadline_s: float = 900.0
    two_read: bool = False
    declared_identity: DeclaredIdentity | None = None
    tau_file: Path | None = None  # a local calibration file; config never opens it

    @property
    def model_host(self) -> str | None:
        """The host name of the model server, lower-case, or None when no server is set."""
        return None if self.model_url is None else urlsplit(self.model_url).hostname

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str],
        *,
        allow_remote: bool = False,
        windows: bool = os.name == "nt",
    ) -> Settings:
        """Build settings from `environ`; a missing or empty variable keeps its default."""
        url = _get(environ, "FOLIO_MODEL_URL")
        if url is not None:
            _check_url(url)
        name = _get(environ, "FOLIO_MODEL_NAME")
        key = _get(environ, "FOLIO_MODEL_API_KEY")
        if key is not None:
            _check_key(key, url)
        tess = _get(environ, "FOLIO_TESSERACT_PATH")
        weights = _get(environ, "FOLIO_WEIGHTS_DIR")
        defaults = cls(weights_dir=Path())
        return cls(
            weights_dir=Path(weights) if weights else default_weights_dir(environ, windows=windows),
            model_url=url,
            model_name=name,
            model_api_key=key,
            tesseract_path=None if tess is None else _absolute_tesseract(Path(tess)),
            host=_host(_get(environ, "FOLIO_HOST"), allow_remote, defaults.host),
            port=_port(_get(environ, "FOLIO_PORT"), defaults.port),
            line_timeout_s=_seconds(environ, "FOLIO_LINE_TIMEOUT_S", defaults.line_timeout_s),
            page_deadline_s=_seconds(environ, "FOLIO_PAGE_DEADLINE_S", defaults.page_deadline_s),
            two_read=_two_read(_get(environ, "FOLIO_TWO_READ")),
            declared_identity=_declared_identity(
                _get(environ, "FOLIO_MODEL_DIGEST"), _get(environ, "FOLIO_MODEL_SERVER")
            ),
            tau_file=_tau_file(_get(environ, "FOLIO_TAU_FILE")),
        )


def _get(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value if value else None


def _has_control(value: str) -> bool:
    return any(unicodedata.category(c) == "Cc" for c in value)


def _check_url(url: str) -> None:
    bad = ConfigError("FOLIO_MODEL_URL must be an http:// or https:// URL with a host name")
    if " " in url or _has_control(url):
        raise bad
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - reading it raises ValueError for an invalid port
    except ValueError:
        raise bad from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise bad


def _check_key(key: str, url: str | None) -> None:
    if _has_control(key):
        raise ConfigError("FOLIO_MODEL_API_KEY must not hold control characters")
    if url is None:
        return
    parts = urlsplit(url)
    if parts.scheme == "http" and not is_loopback(parts.hostname or ""):
        raise ConfigError(
            "FOLIO_MODEL_API_KEY is refused over http:// to a host outside this machine; "
            "use an https:// FOLIO_MODEL_URL"
        )


def _host(value: str | None, allow_remote: bool, default: str) -> str:
    if value is None:
        return default
    try:
        ipaddress.ip_address(value)
    except ValueError:
        if not _HOST_NAME.fullmatch(value):
            raise ConfigError("FOLIO_HOST must be an IP address or a host name") from None
    if not allow_remote and not is_loopback(value):
        raise ConfigError(
            "FOLIO_HOST is not a loopback address; folio serve --allow-remote is needed "
            "to listen beyond this machine"
        )
    return value


def _port(value: str | None, default: int) -> int:
    if value is None:
        return default
    if len(value) <= 5 and _WHOLE_NUMBER.fullmatch(value) and 1 <= int(value) <= 65535:
        return int(value)
    raise ConfigError("FOLIO_PORT must be a whole number from 1 to 65535")


def _seconds(environ: Mapping[str, str], name: str, default: float) -> float:
    value = _get(environ, name)
    if value is None:
        return default
    if _SECONDS.fullmatch(value) and math.isfinite(float(value)) and float(value) > 0:
        return float(value)
    raise ConfigError(f"{name} must be a number of seconds above 0")


def _two_read(value: str | None) -> bool:
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ConfigError("FOLIO_TWO_READ must be 0 or 1")


def _digest(value: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ConfigError("FOLIO_MODEL_DIGEST must be 40 or 64 hexadecimal digits")
    return value.lower()


def _server(value: str) -> tuple[str, str]:
    """(kind, version) from `<kind> <version>`; a version shaped like major.minor[.more] is
    cut to major.minor, any other version is kept as written."""
    parts = value.split(" ")
    if len(parts) != 2 or not _SERVER_KIND.fullmatch(parts[0]) or not parts[1]:
        raise ConfigError(
            "FOLIO_MODEL_SERVER must be <kind> <version> with one space between them, the kind "
            "starting with a lower-case letter or digit, then lower-case letters, digits, '.', "
            "'_' and '-'"
        )
    kind, version = parts
    try:
        return kind, version_major_minor(version)
    except ValueError:
        return kind, version


def _declared_identity(digest: str | None, server: str | None) -> DeclaredIdentity | None:
    checked_digest = None if digest is None else _digest(digest)
    checked_server = None if server is None else _server(server)
    if checked_digest is None and checked_server is None:
        return None
    if checked_server is None:
        raise ConfigError("FOLIO_MODEL_SERVER must be set too when FOLIO_MODEL_DIGEST is set")
    if checked_digest is None:
        raise ConfigError("FOLIO_MODEL_DIGEST must be set too when FOLIO_MODEL_SERVER is set")
    kind, version = checked_server
    return DeclaredIdentity(kind=kind, version=version, digest=checked_digest)


def _tau_file(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        raise ConfigError("FOLIO_TAU_FILE must be an absolute path to a calibration file")
    return path


def is_loopback(host: str) -> bool:
    """True when `host` is `localhost` or an address that stays on this machine."""
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def default_weights_dir(environ: Mapping[str, str], *, windows: bool) -> Path:
    """The per-user folder where model weights and Tesseract data are kept."""
    if windows:
        base = _get(environ, "LOCALAPPDATA")
        if base:
            return Path(base) / "folio-ocr" / "assets"
        return _home() / "AppData" / "Local" / "folio-ocr" / "assets"
    xdg = _get(environ, "XDG_DATA_HOME")
    if xdg and PurePosixPath(xdg).is_absolute():
        return Path(xdg) / "folio-ocr" / "assets"
    return _home() / ".local" / "share" / "folio-ocr" / "assets"


def _home() -> Path:
    try:
        return Path.home()
    except RuntimeError:
        raise ConfigError("cannot find the home folder; set FOLIO_WEIGHTS_DIR") from None


def _absolute_tesseract(value: Path) -> Path:
    if not value.is_absolute():
        raise ConfigError("FOLIO_TESSERACT_PATH must be an absolute path to the tesseract program")
    return value


def _is_file(candidate: Path) -> bool:
    """True for an existing file; a folder that cannot be searched counts as not holding it."""
    try:
        return candidate.is_file()
    except OSError:
        return False


def _runnable(candidate: Path, windows: bool) -> bool:
    return _is_file(candidate) and (windows or os.access(candidate, os.X_OK))


def find_tesseract(
    settings: Settings,
    environ: Mapping[str, str] = os.environ,
    *,
    windows: bool = os.name == "nt",
) -> Path | None:
    """The Tesseract program to run, or None when there is none.

    Order: the configured path, then the absolute folders of PATH, then (on Windows) the
    default install folder. A relative PATH entry, such as the current folder, is never used.
    """
    if settings.tesseract_path is not None:
        return _absolute_tesseract(settings.tesseract_path)
    name = "tesseract.exe" if windows else "tesseract"
    for entry in environ.get("PATH", "").split(os.pathsep):
        if entry and Path(entry).is_absolute() and _runnable(Path(entry) / name, windows):
            return Path(entry) / name
    program_files = environ.get("ProgramFiles") if windows else None
    if program_files and Path(program_files).is_absolute():
        default = Path(program_files) / "Tesseract-OCR" / "tesseract.exe"
        if _is_file(default):
            return default
    return None
