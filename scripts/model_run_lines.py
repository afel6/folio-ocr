"""Per-line measures of model replies against ground-truth lines.

Pure functions: script-run order, tashkeel handling, digit errors, Latin letters inside
Arabic lines, real-word misreads, and pooled summaries over records.
"""

import itertools
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

from folio_ocr import metrics
from folio_ocr.charsets import is_arabic_letter, is_latin_letter
from folio_ocr.normalize import norm_v2

TASHKEEL = frozenset(map(chr, range(0x064B, 0x0653))) | {"ٰ"}
ORDER_CLASSES = (
    "arabic+latin",
    "arabic+1run/european",
    "arabic+1run/arabic-indic",
    "arabic+2runs/european",
    "arabic+2runs/arabic-indic",
)
MIN_ORDER_LINES = 40

_PRIVATE_BASE = 0xE000
_TASHKEEL_TO_PRIVATE = {ord(m): chr(_PRIVATE_BASE + i) for i, m in enumerate(sorted(TASHKEEL))}
_PRIVATE_TO_TASHKEEL = {ord(v): chr(k) for k, v in _TASHKEEL_TO_PRIVATE.items()}


def digit_script(ch: str) -> str | None:
    cp = ord(ch)
    if 0x30 <= cp <= 0x39:
        return "european"
    if 0x660 <= cp <= 0x669 or 0x6F0 <= cp <= 0x6F9:
        return "arabic-indic"
    return None


def digit_runs(text: str) -> tuple[str, ...]:
    """Maximal runs of consecutive digits of one digit script, in text order."""
    runs: list[str] = []
    current = ""
    current_script: str | None = None
    for ch in text:
        script = digit_script(ch)
        if script is not None and script == current_script:
            current += ch
            continue
        if current:
            runs.append(current)
        current, current_script = (ch, script) if script is not None else ("", None)
    if current:
        runs.append(current)
    return tuple(runs)


def order_classes(gt_text: str) -> tuple[str, ...]:
    """Order-check classes a ground-truth line belongs to."""
    if not any(is_arabic_letter(c) for c in gt_text):
        return ()
    if any(is_latin_letter(c) for c in gt_text):
        return ("arabic+latin",)
    runs = digit_runs(gt_text)
    if not runs:
        return ()
    kind = "1run" if len(runs) == 1 else "2runs"
    scripts = {digit_script(r[0]) for r in runs}
    return tuple(f"arabic+{kind}/{s}" for s in ("european", "arabic-indic") if s in scripts)


def _kind(ch: str) -> str | None:
    if is_arabic_letter(ch):
        return "arabic"
    if is_latin_letter(ch):
        return "latin"
    if digit_script(ch) is not None:
        return "digit"
    return None


def script_runs(text: str) -> tuple[tuple[str, str], ...]:
    """(kind, content) runs of the normalised text; neutrals inside a letter run are dropped."""
    runs: list[tuple[str, str]] = []
    kind: str | None = None
    content = ""
    for ch in norm_v2(text):
        k = _kind(ch)
        if k is None:
            if kind == "digit":
                runs.append((kind, content))
                kind, content = None, ""
            continue
        if k == kind:
            content += ch
            continue
        if kind is not None:
            runs.append((kind, content))
        kind, content = k, ch
    if kind is not None:
        runs.append((kind, content))
    return tuple(runs)


def _strong_chars(text: str) -> list[tuple[str, str]]:
    """(kind, character) of every strong character of the normalised text, neutrals dropped."""
    return [(k, ch) for ch in norm_v2(text) if (k := _kind(ch)) is not None]


def _place_run(kind: str, content: str, flat: list[tuple[str, str]],
               taken: frozenset[int]) -> tuple[tuple[int, str] | None, frozenset[int]]:
    """((start, window) of the free reply window closest to the run, or None when none is close,
    and the taken positions including that window)."""
    size = len(content)
    options = (content, content[::-1]) if kind == "digit" else (content,)
    best: tuple[float, int] | None = None
    for start in range(len(flat) - size + 1):
        if flat[start][0] != kind or any(i in taken for i in range(start, start + size)):
            continue
        window = "".join(ch for _, ch in flat[start:start + size])
        rates = [metrics.char_errors(o, window).rate for o in options]
        score = min((r for r in rates if r is not None), default=float("inf"))
        if best is None or score < best[0]:
            best = (score, start)
    if best is None or best[0] > 0.5:
        return None, taken
    start = best[1]
    placed = (start, "".join(ch for _, ch in flat[start:start + size]))
    return placed, taken | frozenset(range(start, start + size))


def order_error(gt_text: str, reply_text: str) -> bool:
    """True when a script run of the ground truth sits in a different place in the reply."""
    flat, taken = _strong_chars(reply_text), frozenset()
    placed = []
    for kind, content in script_runs(gt_text):
        place, taken = _place_run(kind, content, flat, taken)
        placed.append((kind, content, place))
    starts = [p[0] for _, _, p in placed if p is not None]
    if any(b <= a for a, b in itertools.pairwise(starts)):
        return True
    return any(kind == "digit" and len(content) >= 2 and p is not None
               and p[1] == content[::-1] and p[1] != content for kind, content, p in placed)


def with_tashkeel(text: str) -> str:
    """norm_v2 that keeps the short-vowel marks."""
    shielded = text.translate(_TASHKEEL_TO_PRIVATE)
    return norm_v2(shielded).translate(_PRIVATE_TO_TASHKEEL)


def tashkeel_count(text: str) -> int:
    return sum(c in TASHKEEL for c in unicodedata.normalize("NFC", text))


def _digits_only(text: str) -> str:
    return "".join(c for c in norm_v2(text) if digit_script(c) is not None)


def _misreads(gt: str, reply: str, vocabulary: frozenset[str]) -> int:
    left = Counter(norm_v2(reply).split()) - Counter(norm_v2(gt).split())
    return sum(n for word, n in left.items() if word in vocabulary)


def _latin_in_arabic(gt: str, reply: str) -> int:
    if any(is_arabic_letter(c) for c in gt) and not any(is_latin_letter(c) for c in gt):
        return sum(is_latin_letter(c) for c in reply)
    return 0


def line_measures(gt_text: str, reply_text: str, vocabulary: frozenset[str]) -> dict:
    """All per-line measures of one reply against its ground-truth line."""
    digits = metrics.char_errors(_digits_only(gt_text), _digits_only(reply_text))
    kept = metrics.digit_script_preserved(
        unicodedata.normalize("NFC", gt_text), unicodedata.normalize("NFC", reply_text)
    )
    return {
        "cer": metrics.char_errors(norm_v2(gt_text), norm_v2(reply_text)).rate,
        "cer_with_tashkeel": metrics.char_errors(
            with_tashkeel(gt_text), with_tashkeel(reply_text)
        ).rate,
        "tashkeel_gt": tashkeel_count(gt_text),
        "tashkeel_reply": tashkeel_count(reply_text),
        "digit_inserts": digits.insertions,
        "digit_deletes": digits.deletions,
        "latin_in_arabic": _latin_in_arabic(gt_text, reply_text),
        "real_word_misreads": _misreads(gt_text, reply_text, vocabulary),
        "digits_kept": kept.numerator,
        "digits_total": kept.denominator,
        "order_error": order_error(gt_text, reply_text) if order_classes(gt_text) else None,
    }


def pooled(
    records: Iterable[dict],
    key_ref: str = "gt",
    key_hyp: str = "text",
    normalise: Callable[[str], str] = norm_v2,
) -> dict:
    """Character error counts summed over records with a non-empty normalised reference."""
    lines = chars = errors = 0
    for rec in records:
        ref = normalise(rec[key_ref])
        if not ref:
            continue
        counts = metrics.char_errors(ref, normalise(rec.get(key_hyp) or ""))
        lines += 1
        chars += counts.reference_length
        errors += counts.edits
    return {
        "lines": lines,
        "reference_chars": chars,
        "errors": errors,
        "cer": errors / chars if chars else None,
    }


def _order_class_row(rows: Sequence[dict], flag: str) -> dict:
    wrong = [r for r in rows if r["order_error"] is True]
    unflagged = [r for r in wrong if flag not in r["flags"]]
    by_any = [
        r
        for r in wrong
        if not any(f != "low_confidence" for f in r["flags"]) and r["confidence"] is not None
    ]
    status = "ok"
    if len(rows) < MIN_ORDER_LINES:
        status = "short"
    elif unflagged:
        status = "widen"
    return {
        "lines": len(rows),
        "order_errors": len(wrong),
        "unflagged_order_errors": len(unflagged),
        "unflagged_by_any_flag": len(by_any),
        "status": status,
    }


def order_summary(records: Iterable[dict], flag: str = "mixed_order") -> dict:
    """Per-class order-error counts for model-read lines, and a freeze verdict."""
    model = [r for r in records if r["reader"] == "model"]
    classes = {
        name: _order_class_row([r for r in model if name in r["classes"]], flag)
        for name in ORDER_CLASSES
    }
    frozen = all(row["status"] == "ok" for row in classes.values())
    return {
        "classes": classes,
        "verdict": "frozen" if frozen else "stop",
        "min_lines": MIN_ORDER_LINES,
    }


def _share(num: int, den: int) -> float | None:
    return num / den if den else None


def _group_summary(rows: Sequence[dict]) -> dict:
    truncated = [r for r in rows if "truncated" in r["flags"]]
    at_max = [
        r
        for r in truncated
        if r.get("completion_tokens") is not None
        and r.get("max_tokens") is not None
        and r["completion_tokens"] >= r["max_tokens"]
    ]
    kept = sum(r["digits_kept"] for r in rows)
    total = sum(r["digits_total"] for r in rows)
    return {
        "lines": len(rows),
        "cer": pooled(rows)["cer"],
        "cer_with_tashkeel": pooled(rows, normalise=with_tashkeel)["cer"],
        "tashkeel_added_lines": sum(r["tashkeel_reply"] > r["tashkeel_gt"] for r in rows),
        "tashkeel_dropped_lines": sum(r["tashkeel_reply"] < r["tashkeel_gt"] for r in rows),
        "digit_inserts": sum(r["digit_inserts"] for r in rows),
        "digit_deletes": sum(r["digit_deletes"] for r in rows),
        "latin_in_arabic_lines": sum(r["latin_in_arabic"] > 0 for r in rows),
        "latin_in_arabic_letters": sum(r["latin_in_arabic"] for r in rows),
        "real_word_misreads": sum(r["real_word_misreads"] for r in rows),
        "real_word_misread_lines": sum(r["real_word_misreads"] > 0 for r in rows),
        "digits_kept": kept,
        "digits_total": total,
        "digit_script_kept_share": _share(kept, total),
        "truncated_lines": len(truncated),
        "truncated_share": _share(len(truncated), len(rows)),
        "truncated_at_max_tokens": len(at_max),
    }


def measure_summary(records: Iterable[dict]) -> dict:
    """Per script group ("en", "ar") summary of the measures over model-read lines."""
    model = [r for r in records if r["reader"] == "model"]
    groups = sorted({"en", "ar"} | {r["script_group"] for r in model})
    return {g: _group_summary([r for r in model if r["script_group"] == g]) for g in groups}
