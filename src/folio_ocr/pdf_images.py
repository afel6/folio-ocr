"""The largest pixel size that the stream dictionaries of a PDF state, read from the raw file
without decoding anything.

PDFium has no call that returns the size of an image's /SMask or /Mask, or of an image that
only an annotation, a pattern or a Type 3 glyph draws, so the file itself is read here the
way PDFium's parser reads a dictionary: a name may hold #xx escapes, a comment may sit
between tokens, a value may be a reference to a number object, the last of two equal keys
wins, and in a key position every word that is not a name is passed over (a "(" there does
not start a string). The dictionary after every obj keyword in the file ("7 0 obj") is
read, wherever it sits, so no object written in the file can hide another.

The result is incomplete when a size names an object that is not a plain number object in
the file (a missing one, a reference, or any object of a file that has object streams), when
a stream is nested inside a dictionary (PDFium may end it elsewhere), or when reading takes
more steps or bytes than the file's size allows. Not read: a stream that PDFium accepts from
inside a compressed object stream, and a dictionary nested deeper than PDFium's limit after
that limit was lowered by /Length references PDFium followed on the way."""

import re
from typing import NamedTuple

_REGULAR = rb"[^\x00\t\n\x0c\r ()<>\[\]{}/%]"
_WORD_END = rb"(?!" + _REGULAR + rb")"
_GAP = rb"(?:[\x00\t\n\x0c\r ]|%[^\r\n]*)*+"  # white space and comments
_NUM = rb"[0-9+\-.]++" + _WORD_END
_REF = _NUM + _GAP + _NUM + _GAP + rb"R" + _WORD_END

_SKIP_GAP = re.compile(_GAP)
_WORD = re.compile(rb"<<|>>|/" + _REGULAR + rb"*+|" + _REGULAR + rb"++|.", re.DOTALL)
_KEY_SEARCH = re.compile(rb"/|>>|(?<!" + _REGULAR + rb")endobj" + _WORD_END)
_NUMBER = re.compile(_NUM)
_REF_AFTER_NUMBER = re.compile(_GAP + _NUM + _GAP + rb"R" + _WORD_END)
# Objects with nothing nested in them, matched in one go: a reference, a number, a name,
# true/false/null, a string without "(" ")" inside, or a hex string.
_FLAT = (
    rb"(?>"
    + _REF
    + rb"|"
    + _NUM
    + rb"|/"
    + _REGULAR
    + rb"*+|(?:true|false|null)"
    + _WORD_END
    + rb"|\((?:[^()\\]++|\\.)*+\)|<(?!<)[^>]*+>)"
)
_FLAT_ITEMS = rb"(?:" + _GAP + _FLAT + rb")*+"
_SIMPLE_ITEMS = re.compile(_FLAT_ITEMS, re.DOTALL)
_STRING_SPECIAL = re.compile(rb"\\.|[()]", re.DOTALL)
_STREAM_KEYWORD = re.compile(rb"stream" + _WORD_END)
# The keyword obj that ends every object header ("7 0 obj"). The literal comes first so
# the search runs at memory speed.
_OBJ_KEYWORD = rb"obj(?<!" + _REGULAR + rb"obj)" + _WORD_END
_OBJ = re.compile(_OBJ_KEYWORD)
# A plain header's two numbers, read back from just before its obj keyword.
_PLAIN_HEADER = re.compile(
    rb"(?<!" + _REGULAR + rb")([0-9]{1,10})[\x00\t\n\x0c\r ]++[0-9]{1,10}[\x00\t\n\x0c\r ]++\Z"
)
_HEADER_REACH = 32  # bytes before obj that hold a plain header's two numbers
_HEX_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")
_LEADING_NUMBER = re.compile(rb"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)")

_MAX_DEPTH = 64  # PDFium gives up on an object nested deeper than this
_INT_MAX = 2**31 - 1
# The reading budget grows with the file, so it does not depend on the machine's speed.
_BASE_STEPS = 20_000
_BYTES_PER_STEP = 64
_STEPS_PER_REFERENCED_SIZE = 8
_BASE_BYTES = 1 << 20  # bytes of object dictionaries read, beyond twice the file's size
_READS_PER_BYTE = 2
_BYTES_PER_KEYWORD = 32  # what each obj keyword costs the reading budget, at least
_MAX_INT_DIGITS = 10  # longer digit runs are outside 32 bits
_KEYS = (b"Width", b"Height", b"Length")
_FAIL = object()


def _name_pattern(name: bytes) -> bytes:
    """A pattern for `name` with any of its characters written as a #xx escape."""

    def one(char: int) -> bytes:
        digits = b"".join(
            b"[%c%c]" % (d, d | 0x20) if d > 0x39 else b"%c" % d for d in b"%02X" % char
        )
        return b"(?:%s|#%s)" % (re.escape(bytes([char])), digits)

    return b"".join(one(char) for char in name)


# What a key position can skip in one go: a name that is not a size key followed by a simple
# value, a comment, and any text that holds no name, ">>" or endobj word (in a key position
# PDFium passes over every word that is not a name, a "(" included).
_OBJECT_STREAM = re.compile(rb"/" + _name_pattern(b"ObjStm") + _WORD_END)
_SIZE_KEY = rb"(?:" + rb"|".join(_name_pattern(k) for k in _KEYS) + rb")" + _WORD_END
_SIMPLE_VALUE = rb"(?>" + _FLAT + rb"|\[" + _FLAT_ITEMS + _GAP + rb"\])"
_OTHER_ENTRIES = re.compile(
    rb"(?:[^/>e%]++|%[^\r\n]*+|>(?!>)|(?<=" + _REGULAR + rb")e|e(?!ndobj" + _WORD_END + rb")"
    rb"|/(?!" + _SIZE_KEY + rb")" + _REGULAR + rb"++" + _GAP + _SIMPLE_VALUE + rb")*+",
    re.DOTALL,
)


class _Ref(int):
    """The object number of an indirect reference."""


def _number(word: bytes) -> int:
    """A numeric word as PDFium's GetInteger reads it: a real is truncated toward zero and
    saturated; an integer outside 32 bits holds no positive size (0 here)."""
    match = _LEADING_NUMBER.match(word)
    if not match:
        return 0
    text = match.group()
    if b"." in text:
        return int(max(-_INT_MAX - 1, min(_INT_MAX, float(text))))
    digits = text.lstrip(b"+-").lstrip(b"0") or b"0"  # leading zeros never change a number
    if len(digits) > _MAX_INT_DIGITS:
        return 0
    value = -int(digits) if text.startswith(b"-") else int(digits)
    return value if -_INT_MAX - 1 <= value <= _INT_MAX else 0


def _reference(word: bytes) -> object:
    """The size a reference whose object number is `word` gives, as PDFium reads it:
    object 0 names nothing (0), 4294967295 and -1 are no reference at all (_FAIL), and a
    number not written as plain digits is a reference that cannot be read (_Ref(-1))."""
    if word == b"-1":
        return _FAIL
    digits = word.lstrip(b"0") or b"0"
    if not word.isdigit() or len(digits) > _MAX_INT_DIGITS:
        return _Ref(-1)
    number = int(digits)
    if number >= 2**32 - 1:
        return _FAIL if number == 2**32 - 1 else _Ref(-1)
    return _Ref(number) if number else 0


def _key(name: bytes) -> bytes:
    return _HEX_ESCAPE.sub(lambda m: bytes([int(m.group(1), 16)]), name[1:])


def _as_int(value: object) -> int:
    """A size value that is a number, else 0 (PDFium reads any other object as 0)."""
    return value if type(value) is int else 0


def _pixels(width: int, height: int) -> int:
    return width * height if width > 0 and height > 0 else 0


class _CannotTell(Exception):
    """The reader cannot vouch for every size: it ran out of budget, or met a stream nested
    inside a dictionary."""


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.largest = 0  # pixels of the largest stream whose size is given as numbers
        self.by_reference: list[tuple[object, object]] = []  # sizes that name an object
        self.steps_left = _BASE_STEPS + len(data) // _BYTES_PER_STEP
        self.bytes_left = _BASE_BYTES + _READS_PER_BYTE * len(data)

    def spend(self, count: int) -> None:
        """Count `count` bytes read against the file's reading budget. Dictionaries that
        overlap are counted once per obj keyword that starts one."""
        self.bytes_left -= count
        if self.bytes_left < 0:
            raise _CannotTell

    def step(self, steps: int = 1) -> None:
        self.steps_left -= steps
        if self.steps_left < 0:
            raise _CannotTell

    def keep_size(self, width: object, height: object) -> None:
        if isinstance(width, _Ref) or isinstance(height, _Ref):
            self.step(_STEPS_PER_REFERENCED_SIZE)  # bounds the memory these take
            self.by_reference.append((width, height))
        else:
            self.largest = max(self.largest, _pixels(_as_int(width), _as_int(height)))

    def read_word(self, pos: int) -> tuple[bytes, int]:
        """`word`, with the bytes it passes counted against the reading budget."""
        word, end = self.word(pos)
        self.spend(end - pos)
        return word, end

    def word(self, pos: int) -> tuple[bytes, int]:
        """The next word after white space and comments, and the position after it."""
        pos = _SKIP_GAP.match(self.data, pos).end()
        match = _WORD.match(self.data, pos)
        return (match.group(), match.end()) if match else (b"", pos)

    def value(self, pos: int, depth: int) -> tuple[object, int]:
        """One object: an int, a _Ref, None for any other object, or _FAIL where PDFium finds
        no object (`>>` is then left unread)."""
        self.step()
        if depth > _MAX_DEPTH:
            return _FAIL, pos
        word, end = self.word(pos)
        if _NUMBER.fullmatch(word):
            ref = _REF_AFTER_NUMBER.match(self.data, end)
            return (_reference(word), ref.end()) if ref else (_number(word), end)
        if word == b"(":
            return None, self.string_end(end)
        if word == b"<":
            close = self.data.find(b">", end)
            return None, len(self.data) if close < 0 else close + 1
        if word == b"[":
            return None, self.array_end(end, depth)
        if word == b"<<":
            return None, self.dictionary(end, depth)
        if word.startswith(b"/") or word in (b"true", b"false", b"null"):
            return None, end
        return _FAIL, end - len(word) if word == b">>" else end

    def array_end(self, pos: int, depth: int) -> int:
        while True:
            pos = _SIMPLE_ITEMS.match(self.data, pos).end()
            item, pos = self.value(pos, depth + 1)
            if item is _FAIL:
                return pos

    def string_end(self, pos: int) -> int:
        """The position after the ")" that closes a string whose "(" ends at `pos`."""
        level = 0
        for part in _STRING_SPECIAL.finditer(self.data, pos):
            self.step()
            mark = part.group()
            if mark == b")":
                if level == 0:
                    return part.end()
                level -= 1
            elif mark == b"(":
                level += 1
        return len(self.data)

    def dictionary(self, pos: int, depth: int) -> int:
        """Read a dictionary body after "<<" and return the position after it. A dictionary
        followed by `stream` has its size kept."""
        entries: dict[bytes, object] = {}
        while True:
            self.step()
            pos = _OTHER_ENTRIES.match(self.data, pos).end()
            match = _KEY_SEARCH.search(self.data, pos)
            if not match:
                return len(self.data)
            if match.group() == b">>":
                return self.after_dictionary(entries, match.end(), depth)
            if match.group() == b"endobj":
                return self.after_dictionary(entries, match.start(), depth)
            pos = self.entry(entries, match.start(), depth)

    def entry(self, entries: dict, pos: int, depth: int) -> int:
        """Read the name at `pos` and its value, keep a size key's value in `entries`, and
        return the position after them (an empty name takes a value that is dropped)."""
        name, pos = self.word(pos)
        value, pos = self.value(pos, depth + 1)
        key = _key(name) if b"#" in name else name[1:]
        if value is not _FAIL and key in _KEYS:
            entries[key] = value
        return pos

    def after_dictionary(self, entries: dict, pos: int, depth: int) -> int:
        """Keep the size of a dictionary followed by `stream`; its contents are never read.
        A stream nested inside another dictionary is one PDFium may end elsewhere, so the
        reader cannot tell what follows it."""
        start = _SKIP_GAP.match(self.data, pos).end()
        self.spend(start - pos)
        if not _STREAM_KEYWORD.match(self.data, start):
            return pos
        if depth > 1:
            raise _CannotTell
        self.keep_size(entries.get(b"Width"), entries.get(b"Height"))
        return start + len(b"stream")


class Declared(NamedTuple):
    """What the stream dictionaries of a PDF state: the largest /Width x /Height (0 if none),
    and whether every size could be read."""

    largest: int
    complete: bool


def _read_objects(reader: _Reader) -> None:
    """Read the dictionary after every obj keyword in the file, wherever it sits (inside a
    string or stream contents included), so no object can hide another."""
    data = reader.data
    for keyword in _OBJ.finditer(data):
        start = _SKIP_GAP.match(data, keyword.end()).end()
        reader.spend(_BYTES_PER_KEYWORD + start - keyword.end())
        if data.startswith(b"<<", start):
            reader.spend(reader.dictionary(start + 2, 1) - start)


def _header_numbers(data: bytes, obj_at: int) -> set[int] | None:
    """The object numbers the plain header ("17 0 obj") whose obj keyword is at `obj_at` can
    define: a cross-reference offset may point inside the number, so "17" also defines 7.
    None when the words before obj are not two plain numbers."""
    header = _PLAIN_HEADER.search(data, max(0, obj_at - _HEADER_REACH), obj_at)
    if not header:
        return None
    digits = header.group(1)
    return {int(digits[i:]) for i in range(len(digits))}


def _defined_value(reader: _Reader, pos: int) -> int | None:
    """The size an object body at `pos` gives: its number, 0 for any other object, or None
    for a reference (not followed)."""
    first, after_first = reader.read_word(pos)
    if not _NUMBER.fullmatch(first):
        return 0
    second, after_second = reader.read_word(after_first)
    if _NUMBER.fullmatch(second) and reader.read_word(after_second)[0] == b"R":
        return None
    return _number(first)


def _definitions(reader: _Reader, wanted: set[int]) -> dict[int, int | None]:
    """What each wanted object number is defined as in the file: the largest number among its
    definitions, None when one is a reference. An obj keyword without a plain header could
    define any number, so it leaves every number out; so does a missing definition."""
    found: dict[int, int | None] = {}
    for keyword in _OBJ.finditer(reader.data):
        reader.spend(_BYTES_PER_KEYWORD)
        numbers = _header_numbers(reader.data, keyword.start())
        if numbers is None:
            return {}
        defined = numbers & wanted
        if not defined:
            continue
        value = _defined_value(reader, keyword.end())
        for number in defined:
            known = found.get(number, 0)
            found[number] = None if value is None or known is None else max(known, value)
    return found


def largest_declared_image(data: bytes) -> Declared:
    """Read every stream dictionary in `data`. A size given as a reference is read from the
    plain number object it names; the result is incomplete when that object is missing, is a
    reference, or may sit in an object stream, when a stream is nested inside a dictionary,
    or when reading takes more steps or bytes than the file's size allows."""
    reader = _Reader(data)
    try:
        _read_objects(reader)
        refs = {v for size in reader.by_reference for v in size if isinstance(v, _Ref)}
        numbers = _definitions(reader, refs) if refs and not _OBJECT_STREAM.search(data) else {}
    except _CannotTell:
        return Declared(reader.largest, False)

    def resolved(v: object) -> int:
        return (numbers.get(v) or 0) if isinstance(v, _Ref) else _as_int(v)

    by_reference = (_pixels(resolved(w), resolved(h)) for w, h in reader.by_reference)
    largest = max(reader.largest, *by_reference, 0)
    return Declared(largest, all(numbers.get(v) is not None for v in refs))
