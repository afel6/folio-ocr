"""Embedding levels and display order of one line of text, by the Unicode Bidirectional
Algorithm (UAX #9) for a given paragraph level and no explicit embeddings, overrides or
isolates: rules X9 (BN removed), W1-W7, N0-N2, I1-I2, L1 and L2."""

import unicodedata
from bisect import bisect_right
from collections.abc import Sequence

EXPLICIT_CODES = frozenset(range(0x202A, 0x202F)) | frozenset(range(0x2066, 0x206A))
_EXPLICIT_CLASSES = frozenset({"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"})
_MAX_BRACKET_DEPTH = 63

# Bidi_Paired_Bracket (opening, closing) from BidiBrackets.txt (the same in Unicode 15.0.0 and
# 15.1.0)
_PAIRS = (
    (0x0028, 0x0029), (0x005B, 0x005D), (0x007B, 0x007D), (0x0F3A, 0x0F3B), (0x0F3C, 0x0F3D),
    (0x169B, 0x169C), (0x2045, 0x2046), (0x207D, 0x207E), (0x208D, 0x208E), (0x2308, 0x2309),
    (0x230A, 0x230B), (0x2329, 0x232A), (0x2768, 0x2769), (0x276A, 0x276B), (0x276C, 0x276D),
    (0x276E, 0x276F), (0x2770, 0x2771), (0x2772, 0x2773), (0x2774, 0x2775), (0x27C5, 0x27C6),
    (0x27E6, 0x27E7), (0x27E8, 0x27E9), (0x27EA, 0x27EB), (0x27EC, 0x27ED), (0x27EE, 0x27EF),
    (0x2983, 0x2984), (0x2985, 0x2986), (0x2987, 0x2988), (0x2989, 0x298A), (0x298B, 0x298C),
    (0x298D, 0x2990), (0x298F, 0x298E), (0x2991, 0x2992), (0x2993, 0x2994), (0x2995, 0x2996),
    (0x2997, 0x2998), (0x29D8, 0x29D9), (0x29DA, 0x29DB), (0x29FC, 0x29FD), (0x2E22, 0x2E23),
    (0x2E24, 0x2E25), (0x2E26, 0x2E27), (0x2E28, 0x2E29), (0x2E55, 0x2E56), (0x2E57, 0x2E58),
    (0x2E59, 0x2E5A), (0x2E5B, 0x2E5C), (0x3008, 0x3009), (0x300A, 0x300B), (0x300C, 0x300D),
    (0x300E, 0x300F), (0x3010, 0x3011), (0x3014, 0x3015), (0x3016, 0x3017), (0x3018, 0x3019),
    (0x301A, 0x301B), (0xFE59, 0xFE5A), (0xFE5B, 0xFE5C), (0xFE5D, 0xFE5E), (0xFF08, 0xFF09),
    (0xFF3B, 0xFF3D), (0xFF5B, 0xFF5D), (0xFF5F, 0xFF60), (0xFF62, 0xFF63),
)  # fmt: skip

Bracket = tuple[bool, int]  # (is opening, canonical opening bracket of the pair)


def _canonical(cp: int) -> int:
    mapped = unicodedata.normalize("NFD", chr(cp))
    return ord(mapped) if len(mapped) == 1 else cp


_BRACKETS: dict[str, Bracket] = {
    **{chr(o): (True, _canonical(o)) for o, _ in _PAIRS},
    **{chr(c): (False, _canonical(o)) for o, c in _PAIRS},
}


# Bidi_Class of unassigned code points by DerivedBidiClass.txt (the same in Unicode 15.0.0 and
# 15.1.0): its @missing ranges plus BN for reserved default-ignorable code points and
# noncharacters; (first, last, class), sorted, L outside them
_UNASSIGNED = (
    (0x0590, 0x05FF, "R"), (0x0600, 0x07BF, "AL"), (0x07C0, 0x085F, "R"), (0x0860, 0x08FF, "AL"),
    (0x2060, 0x206F, "BN"), (0x20A0, 0x20CF, "ET"), (0xFB1D, 0xFB4F, "R"), (0xFB50, 0xFDCF, "AL"),
    (0xFDD0, 0xFDEF, "BN"), (0xFDF0, 0xFDFF, "AL"), (0xFE70, 0xFEFF, "AL"), (0xFFF0, 0xFFF8, "BN"),
    (0x10800, 0x10CFF, "R"), (0x10D00, 0x10D3F, "AL"), (0x10D40, 0x10EBF, "R"),
    (0x10EC0, 0x10EFF, "AL"), (0x10F00, 0x10F2F, "R"), (0x10F30, 0x10F6F, "AL"),
    (0x10F70, 0x10FFF, "R"), (0x1E800, 0x1EC6F, "R"), (0x1EC70, 0x1ECBF, "AL"),
    (0x1ECC0, 0x1ECFF, "R"), (0x1ED00, 0x1ED4F, "AL"), (0x1ED50, 0x1EDFF, "R"),
    (0x1EE00, 0x1EEFF, "AL"), (0x1EF00, 0x1EFFF, "R"), (0xE0000, 0xE0FFF, "BN"),
)  # fmt: skip
_UNASSIGNED_STARTS = tuple(first for first, _, _ in _UNASSIGNED)


def bidi_class(ch: str) -> str:
    """Bidi_Class of one character; unassigned code points take their DerivedBidiClass default."""
    return unicodedata.bidirectional(ch) or _unassigned_class(ord(ch))


def _unassigned_class(cp: int) -> str:
    if cp & 0xFFFE == 0xFFFE:  # noncharacters U+xxFFFE and U+xxFFFF of every plane
        return "BN"
    i = bisect_right(_UNASSIGNED_STARTS, cp) - 1
    if i >= 0 and cp <= _UNASSIGNED[i][1]:
        return _UNASSIGNED[i][2]
    return "L"


def has_explicit_formatting(text: str) -> bool:
    """True when `text` holds an embedding, override or isolate control (U+202A-202E, U+2066-2069)."""
    return any(ord(ch) in EXPLICIT_CODES for ch in text)


def bracket_kind(ch: str) -> str | None:
    """'open' or 'close' for a paired-bracket character (Bidi_Paired_Bracket_Type), else None."""
    bracket = _BRACKETS.get(ch)
    if bracket is None:
        return None
    return "open" if bracket[0] else "close"


def bracket_pairs(text: str) -> list[tuple[int, int]]:
    """Positions of the bracket pairs of `text` by rule BD16, in order of the opening bracket."""
    return _bracket_pairs(["ON"] * len(text), [_BRACKETS.get(ch) for ch in text])


def levels(text: str, base_level: int) -> tuple[int | None, ...]:
    """Resolved level of each character of `text`; None for characters removed by rule X9."""
    return resolve_levels(
        [bidi_class(ch) for ch in text], base_level, [_BRACKETS.get(ch) for ch in text]
    )


def resolve_levels(
    classes: Sequence[str],
    base_level: int,
    brackets: Sequence[Bracket | None] | None = None,
) -> tuple[int | None, ...]:
    """Resolved levels for a sequence of bidi classes (None where the class is BN)."""
    if isinstance(base_level, bool) or base_level not in (0, 1):
        raise ValueError("base level must be 0 or 1")
    if _EXPLICIT_CLASSES.intersection(classes):
        raise ValueError("explicit embeddings, overrides and isolates are not supported")
    kept = [i for i, cls in enumerate(classes) if cls != "BN"]
    original = [classes[i] for i in kept]
    pairs = [brackets[i] if brackets else None for i in kept]
    direction = "R" if base_level else "L"
    types = _weak_types(original, direction)
    types = _paired_brackets(types, original, pairs, direction)
    types = _neutral_types(types, direction)
    resolved = _whitespace_reset(_implicit_levels(types, base_level), original, base_level)
    out: list[int | None] = [None] * len(classes)
    for i, level in zip(kept, resolved, strict=True):
        out[i] = level
    return tuple(out)


def _weak_types(original: Sequence[str], sos: str) -> list[str]:
    types = list(original)
    for i, t in enumerate(types):  # W1
        if t == "NSM":
            types[i] = types[i - 1] if i else sos
    last = sos
    for i, t in enumerate(types):  # W2, W3
        if t in ("L", "R", "AL"):
            last = t
        elif t == "EN" and last == "AL":
            types[i] = "AN"
    types = ["R" if t == "AL" else t for t in types]
    for i in range(1, len(types) - 1):  # W4
        before, here, after = types[i - 1], types[i], types[i + 1]
        if here == "ES" and before == after == "EN":
            types[i] = "EN"
        elif here == "CS" and before == after and before in ("EN", "AN"):
            types[i] = before
    _terminators(types)  # W5
    types = ["ON" if t in ("ES", "ET", "CS") else t for t in types]  # W6
    last = sos
    for i, t in enumerate(types):  # W7
        if t in ("L", "R"):
            last = t
        elif t == "EN" and last == "L":
            types[i] = "L"
    return types


def _terminators(types: list[str]) -> None:
    """Rule W5: a run of ET next to an EN becomes EN."""
    i = 0
    while i < len(types):
        if types[i] != "ET":
            i += 1
            continue
        end = i
        while end < len(types) and types[end] == "ET":
            end += 1
        if (i > 0 and types[i - 1] == "EN") or (end < len(types) and types[end] == "EN"):
            types[i:end] = ["EN"] * (end - i)
        i = end


def _bracket_pairs(types: Sequence[str], pairs: Sequence[Bracket | None]) -> list[tuple[int, int]]:
    """Rule BD16: opening and closing positions of paired brackets, by opening position."""
    stack: list[tuple[int, int]] = []
    found: list[tuple[int, int]] = []
    for i, bracket in enumerate(pairs):
        if bracket is None or types[i] != "ON":
            continue
        opening, ident = bracket
        if opening:
            if len(stack) == _MAX_BRACKET_DEPTH:
                break
            stack.append((ident, i))
            continue
        for depth in range(len(stack) - 1, -1, -1):
            if stack[depth][0] == ident:
                found.append((stack[depth][1], i))
                del stack[depth:]
                break
    return sorted(found)


def _strong(t: str) -> str | None:
    """Direction a resolved type gives to brackets and neutrals (EN and AN count as R)."""
    if t == "L":
        return "L"
    return "R" if t in ("R", "EN", "AN") else None


def _paired_brackets(
    types: list[str], original: Sequence[str], pairs: Sequence[Bracket | None], embedding: str
) -> list[str]:
    """Rule N0, pair by pair in order of the opening bracket."""
    types = list(types)
    for start, end in _bracket_pairs(types, pairs):
        inside = {_strong(t) for t in types[start + 1 : end]}
        if embedding in inside:
            chosen = embedding
        elif inside - {None}:
            # the opposite direction only when the text before the pair runs that way too
            chosen = next((_strong(t) for t in reversed(types[:start]) if _strong(t)), embedding)
        else:
            continue
        for pos in (start, end):
            types[pos] = chosen
            follow = pos + 1
            while follow < len(types) and original[follow] == "NSM":
                types[follow] = chosen
                follow += 1
    return types


def _neutral_types(types: list[str], embedding: str) -> list[str]:
    """Rules N1 and N2: each run of neutrals takes the direction around it, else the embedding."""
    types = list(types)
    i = 0
    while i < len(types):
        if _strong(types[i]):
            i += 1
            continue
        end = i
        while end < len(types) and not _strong(types[end]):
            end += 1
        before = _strong(types[i - 1]) if i else embedding
        after = _strong(types[end]) if end < len(types) else embedding
        types[i:end] = [before if before == after else embedding] * (end - i)
        i = end
    return types


def _implicit_levels(types: Sequence[str], base_level: int) -> list[int]:
    """Rules I1 and I2."""
    if base_level == 0:
        bump = {"L": 0, "R": 1, "AN": 2, "EN": 2}
    else:
        bump = {"L": 1, "R": 0, "AN": 1, "EN": 1}
    return [base_level + bump[t] for t in types]


def _whitespace_reset(levels: list[int], original: Sequence[str], base_level: int) -> list[int]:
    """Rule L1: separators, and whitespace before them or at the line end, take the base level."""
    levels = list(levels)
    trailing = True
    for i in range(len(original) - 1, -1, -1):
        cls = original[i]
        if cls in ("S", "B"):
            levels[i] = base_level
            trailing = True
        elif cls == "WS" and trailing:
            levels[i] = base_level
        else:
            trailing = False
    return levels


def visual_order(resolved: Sequence[int | None]) -> tuple[int, ...]:
    """Rule L2: positions in display order from left to right, skipping None levels."""
    order = [i for i, level in enumerate(resolved) if level is not None]
    lv = [resolved[i] for i in order]
    if not lv:
        return ()
    lowest_odd = min(level for level in lv if level % 2) if any(v % 2 for v in lv) else None
    if lowest_odd is None:
        return tuple(order)
    for level in range(max(lv), lowest_odd - 1, -1):
        i = 0
        while i < len(order):
            if lv[i] < level:
                i += 1
                continue
            end = i
            while end < len(order) and lv[end] >= level:
                end += 1
            order[i:end], lv[i:end] = order[i:end][::-1], lv[i:end][::-1]
            i = end
    return tuple(order)
