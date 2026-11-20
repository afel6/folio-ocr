"""Error metrics for OCR output: character and word error counts, entity and
digit checks, and line-box matching."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from rapidfuzz.distance import Levenshtein

from folio_ocr.charsets import script_of
from folio_ocr.document import Box, Matrix
from folio_ocr.normalize import norm_v2

Normalizer = Callable[[str], str]


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        return self.edits / self.reference_length if self.reference_length else None

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )


@dataclass(frozen=True)
class Ratio:
    numerator: int
    denominator: int

    @property
    def rate(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None


@dataclass(frozen=True)
class EntityMatch:
    reference: str
    hypothesis: str
    raw_exact: bool
    value_exact: bool


_ZERO = ErrorCounts(0, 0, 0, 0)


def _count_ops(ref: Sequence, hyp: Sequence) -> ErrorCounts:
    ops = Levenshtein.editops(ref, hyp)
    tags = [op.tag for op in ops]
    return ErrorCounts(tags.count("replace"), tags.count("delete"), tags.count("insert"), len(ref))


def char_errors(ref: str, hyp: str) -> ErrorCounts:
    """Substitutions, deletions and insertions between code-point sequences."""
    return _count_ops(ref, hyp)


def _require_reference(normalised: str) -> str:
    """Return the normalised reference, raising ValueError when it is empty."""
    if not normalised:
        raise ValueError("reference is empty after normalisation")
    return normalised


def _pooled(
    pairs: Iterable[tuple[str, str]],
    normalize: Normalizer,
    tokenize: Callable[[str], Sequence],
) -> ErrorCounts:
    total = _ZERO
    for ref, hyp in pairs:
        r, h = _require_reference(normalize(ref)), normalize(hyp)
        total = total + _count_ops(tokenize(r), tokenize(h))
    return total


def _words(text: str) -> list[str]:
    return text.split(" ") if text else []


def cer(pairs: Iterable[tuple[str, str]], normalize: Normalizer = norm_v2) -> ErrorCounts:
    """Character errors pooled over all pairs."""
    return _pooled(pairs, normalize, lambda t: t)


def wer(pairs: Iterable[tuple[str, str]], normalize: Normalizer = norm_v2) -> ErrorCounts:
    """Word errors pooled over all pairs."""
    return _pooled(pairs, normalize, _words)


def _script_tally(text: str) -> tuple[int, int]:
    scripts = [script_of(ch) for ch in text]
    return scripts.count("arabic"), scripts.count("latin")


def _majority(arabic: int, latin: int, fallback: str) -> str:
    if arabic > latin:
        return "arabic"
    if latin > arabic:
        return "latin"
    return fallback


def cer_by_script(
    pairs: Iterable[tuple[str, str]], normalize: Normalizer = norm_v2
) -> dict[str, ErrorCounts]:
    """Character errors split by the script of the reference (or inserted) character.

    Characters with no script (digits, punctuation, spaces) count to the line's
    majority script."""
    normed = [(normalize(r), normalize(h)) for r, h in pairs]
    for r, _ in normed:
        _require_reference(r)
    tallies = [_script_tally(r) for r, _ in normed]
    overall = _majority(
        sum(arabic for arabic, _ in tallies), sum(latin for _, latin in tallies), "latin"
    )
    out = {"arabic": _ZERO, "latin": _ZERO}
    for (ref, hyp), (arabic, latin) in zip(normed, tallies, strict=True):
        line = _majority(arabic, latin, overall)
        for key, counts in _line_script_counts(ref, hyp, line).items():
            out[key] = out[key] + counts
    return out


def _line_script_counts(ref: str, hyp: str, line: str) -> dict[str, ErrorCounts]:
    sub = {"arabic": 0, "latin": 0}
    dele = dict(sub)
    ins = dict(sub)
    length = dict(sub)
    for ch in ref:
        length[script_of(ch) or line] += 1
    for op in Levenshtein.editops(ref, hyp):
        if op.tag == "insert":
            ins[script_of(hyp[op.dest_pos]) or line] += 1
        else:
            target = sub if op.tag == "replace" else dele
            target[script_of(ref[op.src_pos]) or line] += 1
    return {k: ErrorCounts(sub[k], dele[k], ins[k], length[k]) for k in ("arabic", "latin")}


def _check_spans(ref: str, spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    checked = []
    for span in spans:
        try:
            s, e = span
        except (TypeError, ValueError):
            raise ValueError("span must be a (start, end) pair") from None
        if any(type(bound) is not int for bound in (s, e)):
            raise ValueError("span bounds must be integers")
        if not 0 <= s < e <= len(ref):
            raise ValueError("span out of range")
        checked.append((s, e))
    return checked


def _aligned_map(ref: str, hyp: str) -> dict[int, int]:
    """Reference index -> hypothesis index for equal and replaced characters."""
    mapping: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag in ("equal", "replace"):
            for i in range(i1, i2):
                mapping[i] = min(j1 + (i - i1), j2 - 1)
    return mapping


def match_entities(ref: str, hyp: str, spans: Iterable[tuple[int, int]]) -> tuple[EntityMatch, ...]:
    """For each reference span, the aligned hypothesis text and whether it matches."""
    checked = _check_spans(ref, spans)
    mapping = _aligned_map(ref, hyp)
    inserts = [
        (i1, j1, j2) for tag, i1, _i2, j1, j2 in Levenshtein.opcodes(ref, hyp) if tag == "insert"
    ]
    out = []
    for s, e in checked:
        idx = [mapping[i] for i in range(s, e) if i in mapping]
        for i1, j1, j2 in inserts:
            if s < i1 < e:
                idx.extend(range(j1, j2))
        sub = hyp[min(idx) : max(idx) + 1] if idx else ""
        out.append(EntityMatch(ref[s:e], sub, ref[s:e] == sub, norm_v2(ref[s:e]) == norm_v2(sub)))
    return tuple(out)


def entity_exact_match(matches: Iterable[EntityMatch], *, by: str) -> Ratio:
    """Share of entities matched exactly, comparing raw text or normalised value."""
    if by not in ("raw", "value"):
        raise ValueError("by must be 'raw' or 'value'")
    items = list(matches)
    hits = sum(m.raw_exact if by == "raw" else m.value_exact for m in items)
    return Ratio(hits, len(items))


def _digit_script(ch: str) -> str | None:
    cp = ord(ch)
    if 0x30 <= cp <= 0x39:
        return "western"
    if 0x660 <= cp <= 0x669:
        return "arabic_indic"
    if 0x6F0 <= cp <= 0x6F9:
        return "extended"
    return None


def digit_script_preserved(ref: str, hyp: str) -> Ratio:
    """Reference digits that stay in the same digit script in the hypothesis."""
    mapping = _aligned_map(ref, hyp)
    total = kept = 0
    for i, ch in enumerate(ref):
        script = _digit_script(ch)
        if script is None:
            continue
        total += 1
        if i in mapping and _digit_script(hyp[mapping[i]]) == script:
            kept += 1
    return Ratio(kept, total)


def compose(a: Matrix, b: Matrix) -> Matrix:
    """The matrix product a·b: `b` is applied first, then `a`."""
    rows = tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )
    return rows  # type: ignore[return-value]


def map_box(matrix: Matrix, box: Box) -> Box:
    """Bounding box of the four box corners projected through `matrix`."""
    x0, y0, x1, y1 = box
    xs: list[float] = []
    ys: list[float] = []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        u, v, w = (r[0] * x + r[1] * y + r[2] for r in matrix)
        if w <= 0:
            raise ValueError("box maps outside the image plane")
        xs.append(u / w)
        ys.append(v / w)
    return (min(xs), min(ys), max(xs), max(ys))


def gt_line_boxes(
    clean_boxes: Sequence[Box], homography: Matrix, transform: Matrix
) -> tuple[Box, ...]:
    """Clean-render ground-truth boxes mapped into the straightened page image frame."""
    full = compose(transform, homography)
    return tuple(map_box(full, box) for box in clean_boxes)


def _area(b: Box) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes; 0.0 when the union is empty."""
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def match_lines(
    gt: Sequence[Box], pred: Sequence[Box], threshold: float = 0.5
) -> tuple[tuple[int, int], ...]:
    """Largest one-to-one matching of boxes whose IoU is at least `threshold`."""
    cands = []
    for g in gt:
        scored = [(iou(g, p), j) for j, p in enumerate(pred)]
        cands.append([j for v, j in sorted(scored, key=lambda t: (-t[0], t[1])) if v >= threshold])
    owner: dict[int, int] = {}

    def augment(g: int, seen: set[int]) -> bool:
        for j in cands[g]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = g
                return True
        return False

    for g in range(len(gt)):
        augment(g, set())
    return tuple(sorted((g, j) for j, g in owner.items()))


def line_recall(gt: Sequence[Box], pred: Sequence[Box], threshold: float = 0.5) -> Ratio:
    """Share of ground-truth lines matched by a predicted line."""
    return Ratio(len(match_lines(gt, pred, threshold)), len(gt))


def reading_order_errors(gt: Sequence[Box], pred: Sequence[Box], threshold: float = 0.5) -> Ratio:
    """Consecutive matched lines whose predicted order runs backwards."""
    matches = match_lines(gt, pred, threshold)
    errors = sum(1 for a, b in pairwise(matches) if b[1] < a[1])
    return Ratio(errors, max(len(matches) - 1, 0))


def missed_error_rate(lines: Iterable[tuple[str, str, str]], threshold: float = 0.02) -> Ratio:
    """Share of erroneous lines (CER above `threshold`) that were auto accepted."""
    erroneous = accepted = 0
    for ref, hyp, status in lines:
        r, h = _require_reference(norm_v2(ref)), norm_v2(hyp)
        rate = char_errors(r, h).rate
        if rate > threshold:
            erroneous += 1
            accepted += status == "auto_accepted"
    return Ratio(accepted, erroneous)
