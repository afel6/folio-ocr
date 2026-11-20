"""The calibration setup of a model server: which frozen threshold, if any, judges a model line.

A key needs the model's digest and the server's version. They come from the configured server
itself (Ollama's /api/tags and /api/version) and only when it has no such routes from the
administrator's declaration, whose key reads "declared:" in the version slot. A server that
fails to report them, or contradicts a declaration, gets no key, so its lines stay uncalibrated
and are flagged low_confidence."""

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from folio_ocr import calibration, model_http, model_request
from folio_ocr.model_reply import ORDER_RULE

SERVER_KIND = "ollama"
AGGREGATION = {"logprobs": "logprobs:exp-min-logprob", "two_read": "two_read:agreement"}
_DIGEST = re.compile(r"[0-9a-f]{64}")


class Declared(Protocol):
    """A model identity the administrator declares (config.DeclaredIdentity has these)."""

    kind: str
    version: str
    digest: str


@dataclass(frozen=True)
class ServerInfo:
    kind: str  # the server software, e.g. "ollama"
    version: str  # its major.minor
    digest: str  # the configured model's digest
    declared: bool = False  # True when the administrator declared it, the server did not say

    @property
    def version_slot(self) -> str:
        """The key's version part; a declared one never equals a server-reported one."""
        slot = f"{self.kind}-{self.version}"
        return f"declared:{slot}" if self.declared else slot


@dataclass(frozen=True)
class Identity:
    info: ServerInfo | None
    source: Literal["server", "declared", "none"]
    reason: str | None  # one line saying why there is no identity; None when there is one


def _json(reply: model_http.Reply) -> object | None:
    """The reply's JSON value, or None for a failed status or a body that is not JSON."""
    if not 200 <= reply.status < 300:
        return None
    try:
        return json.loads(reply.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):  # nesting too deep to parse
        return None


def _version(reply: model_http.Reply) -> str | None:
    value = _json(reply)
    if not isinstance(value, dict) or not isinstance(value.get("version"), str):
        return None
    try:
        return calibration.version_major_minor(value["version"])
    except ValueError:
        return None


def _digest(reply: model_http.Reply, model_name: str) -> str | None:
    """The digest of the one listed model named exactly `model_name`, else None."""
    value = _json(reply)
    models = value.get("models") if isinstance(value, dict) else None
    if not isinstance(models, list):
        return None
    found = [
        entry.get("digest")
        for entry in models
        if isinstance(entry, dict) and model_name in (entry.get("name"), entry.get("model"))
    ]
    if not found or any(d != found[0] for d in found):
        return None
    digest = found[0]
    return digest if isinstance(digest, str) and _DIGEST.fullmatch(digest) else None


def _unreadable(error: model_http.ModelCallError) -> str:
    return f"the server's model identity could not be read: {error}"


_NO_IDENTITY = (
    "the server reports no model identity; set FOLIO_MODEL_DIGEST and FOLIO_MODEL_SERVER "
    "to declare it"
)
_DIFFERS = "FOLIO_MODEL_DIGEST or FOLIO_MODEL_SERVER differs from what the server reports"
_NONE = "none"


def _refuse(reason: str) -> Identity:
    return Identity(None, _NONE, reason)


def _from_declaration(declared: Declared | None) -> Identity:
    if declared is None:
        return _refuse(_NO_IDENTITY)
    info = ServerInfo(declared.kind, declared.version, declared.digest, declared=True)
    return Identity(info, "declared", None)


def _from_replies(
    version_reply: model_http.Reply,
    tags_reply: model_http.Reply,
    model_name: str,
    declared: Declared | None,
) -> Identity:
    version = _version(version_reply)
    if version is None:
        return _refuse("the server reported a version this tool cannot read")
    digest = _digest(tags_reply, model_name)
    if digest is None:
        return _refuse(
            "the server does not list FOLIO_MODEL_NAME exactly once with a sha256 digest"
        )
    if declared is not None and (
        declared.kind != SERVER_KIND or declared.version != version or declared.digest != digest
    ):
        return _refuse(_DIFFERS)
    return Identity(ServerInfo(SERVER_KIND, version, digest), "server", None)


def identify(
    url: str,
    api_key: str | None,
    model_name: str,
    timeout_s: float,
    declared: Declared | None = None,
    *,
    get: Callable[..., model_http.Reply] = model_http.get,
) -> Identity:
    """Where the model's digest and the server's version come from: the server's own identity
    routes when it has them, else the administrator's declaration; never a mix of the two."""
    version_target = model_http.native_target(url, api_key, "version")
    tags_target = model_http.native_target(url, api_key, "tags")
    if version_target is None or tags_target is None:
        return _from_declaration(declared)
    try:
        version_reply = get(version_target, timeout_s)
        tags_reply = get(tags_target, timeout_s)
    except model_http.ModelCallError as error:
        return _refuse(_unreadable(error))
    statuses = (version_reply.status, tags_reply.status)
    if statuses == (404, 404):
        return _from_declaration(declared)
    if not all(200 <= status < 300 for status in statuses):
        return _refuse(
            f"the server answered its model identity routes with HTTP {statuses[0]}/{statuses[1]}"
        )
    return _from_replies(version_reply, tags_reply, model_name, declared)


def server_info(
    url: str,
    api_key: str | None,
    model_name: str,
    timeout_s: float,
    *,
    get: Callable[..., model_http.Reply] = model_http.get,
    declared: Declared | None = None,
) -> ServerInfo | None:
    """What identify finds, or None when there is no key."""
    return identify(url, api_key, model_name, timeout_s, declared, get=get).info


def model_setup(
    info: ServerInfo | None, model_name: str, method: str, page_preprocessing: str
) -> calibration.Setup | None:
    """The setup that judges a line scored by `method` ("logprobs" or "two_read") on a page
    prepared as `page_preprocessing` (the OCR setup's preprocessing hash, or the text-layer
    render's); None when there is no key."""
    if info is None or method not in AGGREGATION:
        return None
    constants = {
        "page": page_preprocessing,
        **model_request.crop_constants(),
        "order_rule": ORDER_RULE,
    }
    try:
        setup = calibration.Setup(
            reader="model",
            architecture=model_name,
            weights_sha256=info.digest,
            aggregation=AGGREGATION[method],
            version=info.version_slot,
            arguments=model_request.arguments(),
            preprocessing=calibration.preprocessing_hash(constants),
        )
        setup.key.encode("utf-8")
    except ValueError:  # "|", a line break, or text that is not UTF-8 cannot form a key
        return None
    return setup


def matching_keys(info: ServerInfo | None, model_name: str, keys: Iterable[str]) -> tuple[str, ...]:
    """The keys that hold this model, digest, server version slot and request constants (for
    any scoring method and page preparation); none without a server identity."""
    if info is None:
        return ()
    head = ["model", model_name, info.digest]
    tail = [info.version_slot, " ".join(model_request.arguments())]
    matches = []
    for key in keys:
        parts = key.split("|")
        if len(parts) == 7 and parts[:3] == head and parts[4:6] == tail:
            matches.append(key)
    return tuple(matches)


def is_calibrated(
    info: ServerInfo | None, model_name: str, table: Mapping[str, float] | None = None
) -> bool:
    """True when the threshold table holds a tau for this model, digest, server version slot
    and request constants (for any scoring method and page preparation)."""
    return bool(matching_keys(info, model_name, calibration.TAUS if table is None else table))
