"""Framing between folio and its file reader process."""

import io
import struct

import pytest

from folio_ocr import pdf_client, pdf_frames
from folio_ocr.pdf_frames import Ended


def framed(body: bytes) -> io.BytesIO:
    return io.BytesIO(struct.pack(">I", len(body)) + body)


def test_header_and_data_round_trip():
    pipe = io.BytesIO()
    pdf_frames.write(pipe, {"op": "open", "n": 1}, b"abc")
    pipe.seek(0)
    assert pdf_frames.read_header(pipe, 100) == {"op": "open", "n": 1}
    assert pdf_frames.read_data(pipe, 3) == b"abc"


def test_empty_data_round_trips():
    pipe = io.BytesIO()
    pdf_frames.write(pipe, {}, b"")
    pipe.seek(0)
    assert pdf_frames.read_header(pipe, 10) == {}
    assert pdf_frames.read_any(pipe, 0) == b""


def test_length_over_the_limit_is_refused_before_the_body_is_read():
    class Spy(io.BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.reads = []

        def readinto(self, buf):
            self.reads.append(len(buf))
            return super().readinto(buf)

    pipe = Spy(struct.pack(">I", 11) + b"x" * 11)
    with pytest.raises(Ended, match="too long"):
        pdf_frames.read_header(pipe, 10)
    assert pipe.reads == [4]


def test_length_equal_to_the_limit_is_read():
    assert pdf_frames.read_header(framed(b'{"a":1}'), 7) == {"a": 1}


def test_wrong_data_length_is_refused():
    pipe = io.BytesIO()
    pdf_frames.write(pipe, {}, b"abcd")
    pipe.seek(0)
    pdf_frames.read_header(pipe, 10)
    with pytest.raises(Ended, match="wrong length"):
        pdf_frames.read_data(pipe, 3)


def test_read_any_refuses_over_the_limit():
    with pytest.raises(Ended, match="too long"):
        pdf_frames.read_any(framed(b"abcd"), 3)


@pytest.mark.parametrize("body", [b"nope", b"\xff\xfe", b"[" * 100000])
def test_body_that_is_not_json_is_refused(body):
    with pytest.raises(Ended, match="not JSON"):
        pdf_frames.read_header(framed(body), 10**6)


@pytest.mark.parametrize("body", [b"[1]", b'"x"', b"3", b"null", b"true"])
def test_json_that_is_not_an_object_is_refused(body):
    with pytest.raises(Ended, match="not a JSON object"):
        pdf_frames.read_header(framed(body), 100)


@pytest.mark.parametrize("raw", [b"", b"\x00\x00", b"\x00\x00\x00\x05ab"])
def test_end_of_file_inside_a_frame_raises_ended(raw):
    with pytest.raises(Ended):
        pdf_frames.read_header(io.BytesIO(raw), 100)


def test_end_of_file_inside_data_raises_ended():
    with pytest.raises(Ended):
        pdf_frames.read_data(io.BytesIO(struct.pack(">I", 5) + b"ab"), 5)


def test_header_of_exactly_the_reader_limit_is_accepted():
    limit = pdf_client._HEADER_LIMIT
    body = b'{"k":"' + b"a" * (limit - 8) + b'"}'
    assert len(body) == limit
    assert pdf_frames.read_header(framed(body), limit) == {"k": "a" * (limit - 8)}
