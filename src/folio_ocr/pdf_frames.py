"""Messages between folio and its file reader process: each message is a 4-byte big-endian
length and that many bytes. A request or reply is one JSON message, followed by one bytes
message when it carries a file or pixels. Nothing is pickled and nothing touches the disk."""

import json
import struct
from typing import BinaryIO

_LENGTH = struct.Struct(">I")


class Ended(Exception):
    """The other side closed the pipe or sent a message this side does not accept."""


def write(stream: BinaryIO, header: dict, data: bytes | None = None) -> None:
    body = json.dumps(header, separators=(",", ":")).encode("ascii")
    stream.write(_LENGTH.pack(len(body)) + body)
    if data is not None:
        stream.write(_LENGTH.pack(len(data)))
        stream.write(data)
    stream.flush()


def _exactly(stream: BinaryIO, size: int) -> bytearray:
    out = bytearray(size)
    view = memoryview(out)
    filled = 0
    while filled < size:
        got = stream.readinto(view[filled:])
        if not got:
            raise Ended("the pipe closed")
        filled += got
    return out


def _length(stream: BinaryIO) -> int:
    return _LENGTH.unpack(_exactly(stream, _LENGTH.size))[0]


def read_header(stream: BinaryIO, limit: int) -> dict:
    """The next JSON message, refused when longer than `limit` bytes or not a JSON object."""
    size = _length(stream)
    if size > limit:
        raise Ended("a message is too long")
    try:
        header = json.loads(_exactly(stream, size).decode("ascii"))
    except (UnicodeDecodeError, ValueError, RecursionError) as err:
        raise Ended("a message is not JSON") from err
    if not isinstance(header, dict):
        raise Ended("a message is not a JSON object")
    return header


def read_data(stream: BinaryIO, size: int) -> bytearray:
    """The next bytes message, which must be exactly `size` bytes long."""
    if _length(stream) != size:
        raise Ended("a bytes message has the wrong length")
    return _exactly(stream, size)


def read_any(stream: BinaryIO, limit: int) -> bytearray:
    """The next bytes message of at most `limit` bytes (the reader side, from folio)."""
    size = _length(stream)
    if size > limit:
        raise Ended("a bytes message is too long")
    return _exactly(stream, size)
