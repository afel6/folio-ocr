"""The local tau file: parsing, refusals, self-test matching and shadowing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from folio_ocr import calibration, model_taus
from folio_ocr.errors import ConfigError
from folio_ocr.model_taus import SelfTestReply, TauEntry

KEY = "model|m|" + "a" * 64 + "|logprobs:exp-min-logprob|ollama-0.40|args|hash"


def valid() -> dict:
    return {
        "schema": "folio-taus/1",
        "entries": {
            KEY: {
                "tau": 0.8,
                "method": "logprobs",
                "bad_lines": 200,
                "documents": 40,
                "dev_bound": 0.09,
                "dev_manifest_sha256": "b" * 64,
                "self_test": [
                    {"line": "en", "text": "Hello", "score": 0.99},
                    {"line": "ar", "text": "marhaba", "score": 0.76},
                ],
                "drift_tolerance": 0.02,
            }
        },
    }


def dump(data: object) -> bytes:
    return json.dumps(data).encode("utf-8")


def mutated(edit) -> bytes:
    data = copy.deepcopy(valid())
    edit(data, data["entries"][KEY])
    return dump(data)


def test_valid_file_is_parsed() -> None:
    raw = dump(valid())
    local = model_taus.parse(raw)
    assert local.sha256 == hashlib.sha256(raw).hexdigest()
    entry = local.entries[KEY]
    assert (entry.key, entry.tau, entry.method) == (KEY, 0.8, "logprobs")
    assert (entry.bad_lines, entry.documents, entry.dev_bound) == (200, 40, 0.09)
    assert entry.dev_manifest_sha256 == "b" * 64
    assert entry.drift_tolerance == 0.02
    assert [r.line for r in entry.self_test] == ["en", "ar"]
    assert entry.self_test[1] == SelfTestReply("ar", "marhaba", 0.76)
    with pytest.raises(TypeError):
        local.entries["x"] = entry  # type: ignore[index]


def test_self_test_is_stored_en_first() -> None:
    raw = mutated(lambda d, e: e["self_test"].reverse())
    assert [r.line for r in model_taus.parse(raw).entries[KEY].self_test] == ["en", "ar"]


@pytest.mark.parametrize("tau", [0, 1, 0.0, 1.0])
def test_tau_bounds_are_accepted(tau) -> None:
    assert model_taus.parse(mutated(lambda d, e: e.update(tau=tau))).entries[KEY].tau == tau


def test_two_read_allows_null_score() -> None:
    def edit(d, e) -> None:
        e["method"] = "two_read"
        e["self_test"][0]["score"] = None

    entry = model_taus.parse(mutated(edit)).entries[KEY]
    assert entry.method == "two_read" and entry.self_test[0].score is None


def test_extra_fields_are_ignored() -> None:
    def edit(d, e) -> None:
        d["note"] = 1
        e["extra"] = [1]
        e["self_test"][0]["more"] = "x"

    assert KEY in model_taus.parse(mutated(edit)).entries


def test_entries_keep_file_order() -> None:
    data = valid()
    other = "model|other"
    data["entries"][other] = copy.deepcopy(data["entries"][KEY])
    assert list(model_taus.parse(dump(data)).entries) == [KEY, other]


def _missing(field: str):
    return lambda d, e: e.pop(field)


FIELDS = [
    "tau",
    "method",
    "bad_lines",
    "documents",
    "dev_bound",
    "dev_manifest_sha256",
    "self_test",
    "drift_tolerance",
]

REPEATED_TOP = b'{"schema":"folio-taus/1","schema":"folio-taus/1","entries":{}}'
REPEATED_ENTRIES = b'{"schema":"folio-taus/1","entries":{"model|a":{},"model|a":{}}}'
REPEATED_INSIDE = b'{"schema":"folio-taus/1","entries":{"model|a":{"tau":0.5,"tau":0.6}}}'
DEEP = b"[" * 6000 + b"]" * 6000
DEEP_BODY = (
    b'{"schema":"folio-taus/1","entries":{"model|a":{"tau":' + b"[" * 6000 + b"]" * 6000 + b"}}}"
)

REFUSED = {
    "invalid utf8": b"\xff\xfe{}",
    "bom": b"\xef\xbb\xbf" + dump(valid()),
    "not json": b"nope",
    "top list": b"[]",
    "schema missing": dump({"entries": {}}),
    "schema v2": dump({"schema": "folio-taus/2", "entries": {}}),
    "entries missing": dump({"schema": "folio-taus/1"}),
    "entries list": dump({"schema": "folio-taus/1", "entries": []}),
    "key not model": dump({"schema": "folio-taus/1", "entries": {"onnxtr|x": {}}}),
    "repeat top": REPEATED_TOP,
    "repeat entries": REPEATED_ENTRIES,
    "repeat inside": REPEATED_INSIDE,
    "nan tau": dump(valid()).replace(b"0.8", b"NaN"),
    "inf dev_bound": dump(valid()).replace(b"0.09", b"Infinity"),
    "neg inf drift": dump(valid()).replace(b"0.02", b"-Infinity"),
    "tau true": mutated(lambda d, e: e.update(tau=True)),
    "bad_lines true": mutated(lambda d, e: e.update(bad_lines=True)),
    "documents true": mutated(lambda d, e: e.update(documents=True)),
    "dev_bound true": mutated(lambda d, e: e.update(dev_bound=True)),
    "drift true": mutated(lambda d, e: e.update(drift_tolerance=True)),
    "tau high": mutated(lambda d, e: e.update(tau=1.0000001)),
    "tau low": mutated(lambda d, e: e.update(tau=-0.1)),
    "tau huge": dump(valid()).replace(b"0.8", b"1e999"),
    "tau huge int": dump(valid()).replace(b"0.8", b"1" + b"0" * 400),
    "bad_lines 199": mutated(lambda d, e: e.update(bad_lines=199)),
    "documents 39": mutated(lambda d, e: e.update(documents=39)),
    "bad_lines fraction": mutated(lambda d, e: e.update(bad_lines=200.5)),
    "bad_lines whole float": mutated(lambda d, e: e.update(bad_lines=200.0)),
    "method median": mutated(lambda d, e: e.update(method="median")),
    "method number": mutated(lambda d, e: e.update(method=5)),
    "sha number": mutated(lambda d, e: e.update(dev_manifest_sha256=5)),
    "entry not dict": dump({"schema": "folio-taus/1", "entries": {"model|a": []}}),
    "self_test not list": mutated(lambda d, e: e.update(self_test={})),
    "self_test item not dict": mutated(lambda d, e: e["self_test"].append(5)),
    "no ar": mutated(lambda d, e: e["self_test"].pop()),
    "en twice": mutated(lambda d, e: e["self_test"][1].update(line="en")),
    "line fr": mutated(lambda d, e: e["self_test"][1].update(line="fr")),
    "line number": mutated(lambda d, e: e["self_test"][1].update(line=1)),
    "three lines": mutated(
        lambda d, e: e["self_test"].append({"line": "en", "text": "", "score": 1})
    ),
    "null score logprobs": mutated(lambda d, e: e["self_test"][0].update(score=None)),
    "score string": mutated(lambda d, e: e["self_test"][0].update(score="x")),
    "score bool": mutated(lambda d, e: e["self_test"][0].update(score=True)),
    "score missing": mutated(lambda d, e: e["self_test"][0].pop("score")),
    "text number": mutated(lambda d, e: e["self_test"][0].update(text=5)),
    "text missing": mutated(lambda d, e: e["self_test"][0].pop("text")),
    "deep": DEEP,
    "deep body": DEEP_BODY,
}
for _field in FIELDS:
    REFUSED[f"missing {_field}"] = mutated(_missing(_field))


@pytest.mark.parametrize("name", list(REFUSED))
def test_refusal(name: str) -> None:
    with pytest.raises(ConfigError) as caught:
        model_taus.parse(REFUSED[name])
    message = str(caught.value)
    assert message.startswith("FOLIO_TAU_FILE")
    assert message.splitlines() == [message]


def test_refusal_hides_key_and_path(tmp_path: Path) -> None:
    secret = "model|SECRETNAME"
    data = valid()
    data["entries"][secret] = {"tau": 5}
    path = tmp_path / "SECRETDIR_taus.json"
    path.write_bytes(dump(data))
    with pytest.raises(ConfigError) as caught:
        model_taus.load(path)
    assert "SECRETNAME" not in str(caught.value)
    assert "SECRETDIR" not in str(caught.value)
    assert str(path) not in str(caught.value)


def test_load_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "taus.json"
    path.write_bytes(dump(valid()))
    assert KEY in model_taus.load(path).entries


@pytest.mark.parametrize("kind", ["missing", "folder"])
def test_load_unreadable(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "nothing" if kind == "missing" else tmp_path
    with pytest.raises(ConfigError) as caught:
        model_taus.load(path)
    assert str(caught.value) == "FOLIO_TAU_FILE cannot be read"


def entry(method: str = "logprobs") -> TauEntry:
    data = valid()
    data["entries"][KEY]["method"] = method
    return model_taus.parse(dump(data)).entries[KEY]


def replies(en: float | None = 0.99, ar: float | None = 0.76, text_en: str = "Hello"):
    return [SelfTestReply("en", text_en, en), SelfTestReply("ar", "marhaba", ar)]


def test_matches_within_tolerance() -> None:
    assert model_taus.matches(entry(), replies(0.99, 0.76))
    assert model_taus.matches(entry(), replies(0.99 + 0.01, 0.76 - 0.01))


def test_matches_at_tolerance_edge() -> None:
    e = entry()
    exact = TauEntry(**{**e.__dict__, "drift_tolerance": 0.25})
    assert model_taus.matches(exact, replies(0.99 - 0.25, 0.76))
    assert not model_taus.matches(exact, replies(0.99 - 0.25 - 1e-9, 0.76))
    assert not model_taus.matches(exact, replies(0.99, 0.76 + 0.25 + 1e-9))


def test_matches_text_must_be_exact() -> None:
    assert not model_taus.matches(entry(), replies(text_en="Hello "))
    assert not model_taus.matches(entry("two_read"), replies(text_en="hello"))


def test_matches_logprobs_needs_scores() -> None:
    assert not model_taus.matches(entry(), replies(en=None))


def test_matches_two_read_ignores_scores() -> None:
    assert model_taus.matches(entry("two_read"), replies(None, 0.0))
    assert model_taus.matches(entry("two_read"), replies(5.0, None))


def test_matches_missing_line() -> None:
    assert not model_taus.matches(entry(), replies()[:1])
    assert not model_taus.matches(entry("two_read"), replies()[:1])
    assert not model_taus.matches(entry("two_read"), [])


def test_shadowed_lists_package_keys_in_file_order() -> None:
    data = valid()
    for name in ("model|b", "model|c"):
        data["entries"][name] = copy.deepcopy(data["entries"][KEY])
    local = model_taus.parse(dump(data))
    package = {"model|c": 0.5, KEY: 0.6, "other": 0.1}
    assert model_taus.shadowed(local, package) == (KEY, "model|c")
    assert model_taus.shadowed(local, {}) == ()


def test_package_table_is_untouched(tmp_path: Path) -> None:
    before = calibration.table_sha256()
    model_taus.parse(dump(valid()))
    assert calibration.table_sha256() == before
    assert KEY not in calibration.TAUS


@pytest.mark.parametrize("field", ["dev_bound", "drift_tolerance", "tau"])
def test_an_overflowing_number_is_refused(field) -> None:
    raw = dump(valid()).replace(f'"{field}": '.encode(), f'"{field}": 1e999, "x{field}": '.encode())
    with pytest.raises(ConfigError, match="^FOLIO_TAU_FILE "):
        model_taus.parse(raw)


@pytest.mark.parametrize("score", ["1e999", "1" + "0" * 400])
def test_an_overflowing_score_is_refused(score) -> None:
    raw = dump(valid()).replace(b'"score": 0.99', f'"score": {score}'.encode())
    with pytest.raises(ConfigError, match="^FOLIO_TAU_FILE "):
        model_taus.parse(raw)


def test_a_two_read_line_without_a_score_is_refused() -> None:
    def edit(_data, entry):
        entry["method"] = "two_read"
        del entry["self_test"][0]["score"]

    with pytest.raises(ConfigError, match="no score"):
        model_taus.parse(mutated(edit))
