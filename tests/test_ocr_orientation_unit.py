"""Page orientation and word finding, tested with doubles for the detector and recogniser."""

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from folio_ocr import onnxtr_models, orientation
from folio_ocr.geometry import straighten
from folio_ocr.ocr_words import Word, _ink_box, find_words, group_lines
from folio_ocr.onnxtr_models import Detection
from folio_ocr.orientation import page_orientation
from folio_ocr.tesseract import Osd, TesseractFailure

MASK = np.zeros((1, 1), dtype=np.uint8)

# A page whose only dark pixel is its top-left one: where that pixel lands says how it was turned.
IMAGE = Image.new("L", (100, 60), 255)
IMAGE.putpixel((0, 0), 0)


def corner(pixels):
    row, col = np.argwhere(np.asarray(pixels)[:, :, 0] == 0)[0]
    return (row == 0, col == 0)


QUARTER = {corner(np.asarray(straighten(IMAGE, q)[0].convert("RGB"))): q for q in (0, 90, 180, 270)}
assert len(QUARTER) == 4

WIDE = np.array([[0.1, 0.1, 0.9, 0.2]])
TALL = np.array([[0.1, 0.1, 0.2, 0.9]])
# A grid of digits: every box about square, so the page has no long word.
DIGITS = np.array([[0.1 * k, 0.1, 0.1 * k + 0.06, 0.2] for k in range(1, 8)])


class Doubles:
    """A detector and recogniser that answer by which way up they are shown the page."""

    def __init__(self, monkeypatch, first, scores, skew=0, found=None):
        self.detected, self.read = [], []
        self.found = found or {}
        self.first = first
        self.scores = scores
        monkeypatch.setattr(onnxtr_models, "detect", self.detect)
        monkeypatch.setattr(onnxtr_models, "recognise", self.recognise)
        monkeypatch.setattr(orientation, "_skew", lambda detection: self._skew(detection, skew))
        self.skewed = []

    def _skew(self, detection, value):
        self.skewed.append(detection)
        return value

    def detect(self, models, rgb):
        quarter = QUARTER[corner(rgb)]
        self.detected.append(quarter)
        boxes = self.first if quarter == 0 else self.found.get(quarter, self.first)
        self.found[quarter] = Detection(boxes, MASK) if isinstance(boxes, np.ndarray) else boxes
        return self.found[quarter]

    def recognise(self, models, rgb, boxes):
        quarter = QUARTER[corner(rgb)]
        self.read.append((quarter, np.asarray(boxes)))
        scores = self.scores[quarter]
        return [("w", s, False) for s in scores]


def run(monkeypatch, first, scores, skew=0, **kw):
    doubles = Doubles(monkeypatch, first, scores, skew, **kw)
    return doubles, page_orientation(None, IMAGE)


def test_no_words_means_upright_and_certain(monkeypatch):
    doubles, result = run(monkeypatch, np.empty((0, 4)), {})
    assert (result.angle, result.uncertain) == (0.0, False)
    assert result.detection.boxes.shape == (0, 4)
    assert doubles.read == [] and doubles.skewed == []


def test_wide_words_choose_between_upright_and_upside_down(monkeypatch):
    doubles, result = run(monkeypatch, WIDE, {0: [0.9], 180: [0.2]})
    assert (result.angle, result.uncertain) == (0.0, False)
    assert [q for q, _ in doubles.read] == [0, 180]
    _, result = run(monkeypatch, WIDE, {0: [0.2], 180: [0.9]})
    assert result.angle == 180.0


def test_tall_words_choose_between_the_two_sideways_ways(monkeypatch):
    doubles, result = run(monkeypatch, TALL, {90: [0.2], 270: [0.9]})
    assert result.angle == 270.0
    assert [q for q, _ in doubles.read] == [90, 270]
    _, result = run(monkeypatch, TALL, {90: [0.9], 270: [0.2]})
    assert result.angle == 90.0


def test_word_shape_is_judged_in_pixels_not_page_fractions():
    # On a 400 x 60 page this box is 80 px wide and 12 px tall: a long wide word, although it
    # covers equal fractions of the page width and height.
    word = np.array([[0.1, 0.1, 0.3, 0.3]])
    assert orientation._sideways(word, 400, 60) is False
    assert orientation._sideways(word, 60, 400) is True


def test_half_the_long_box_area_in_tall_boxes_is_not_sideways(monkeypatch):
    # On the 100 x 60 page: 10 x 60 px tall and 60 x 10 px wide, both 600 px and both long.
    tall, wide = [0.0, 0.0, 0.1, 1.0], [0.0, 0.5, 0.6, 2 / 3]
    doubles, _ = run(monkeypatch, np.array([tall, wide]), {0: [0.5, 0.5], 180: [0.5, 0.5]})
    assert doubles.detected == [0, 180]
    bigger = [0.0, 0.0, 0.12, 1.0]  # 12 x 60 px
    doubles, _ = run(monkeypatch, np.array([bigger, wide]), {90: [0.5, 0.5], 270: [0.5, 0.5]})
    assert doubles.detected[1:] == [90, 270]


def test_short_boxes_do_not_vote_on_sideways():
    # 50 x 18 px and 12 x 30 px: neither side reaches three times the other.
    short = np.array([[0.0, 0.0, 0.5, 0.3], [0.5, 0.5, 0.62, 1.0]])
    assert orientation._sideways(short, 100, 60) is None
    long_wide = np.array([[0.0, 0.0, 0.6, 0.3]])  # 60 x 18 px: 3.33 times
    assert orientation._sideways(np.vstack([short, long_wide]), 100, 60) is False
    at_ratio = np.array([[0.0, 0.0, 0.54, 0.3]])  # 54 x 18 px: exactly three times is long
    assert orientation._sideways(at_ratio, 100, 60) is False


def test_a_page_of_single_characters_is_read_all_four_ways(monkeypatch):
    # A grid of digits: every box about square, upright digits taller than wide.
    digits = np.array([[0.1 * k, 0.1, 0.1 * k + 0.06, 0.2] for k in range(1, 8)])
    scores = {0: [0.45] * 7, 90: [0.2] * 7, 180: [0.4] * 7, 270: [0.95] * 7}
    doubles, result = run(monkeypatch, digits, scores)
    assert doubles.detected == [0, 90, 180, 270]
    assert [q for q, _ in doubles.read] == [0, 90, 180, 270]
    assert result.angle == 270.0 and result.uncertain is False  # 0.95 - 0.45 >= the margin
    scores = {0: [0.9] * 7, 90: [0.2] * 7, 180: [0.85] * 7, 270: [0.3] * 7}
    _, result = run(monkeypatch, digits, scores)
    assert result.angle == 0.0 and result.uncertain is True  # the runner-up is within it


def test_the_vote_is_the_upper_quartile_of_word_scores(monkeypatch):
    # upper quartile 0.65 against 0.6 (the mean and the median favour the other way up)
    scores = {0: [0.2, 0.4, 0.6, 0.8], 180: [0.6] * 4}
    _, result = run(monkeypatch, np.repeat(WIDE, 4, axis=0), scores)
    assert result.angle == 0.0
    assert result.uncertain is True  # margin 0.05


def test_a_tie_keeps_the_first_of_the_pair_and_is_uncertain(monkeypatch):
    _, result = run(monkeypatch, WIDE, {0: [0.7], 180: [0.7]})
    assert (result.angle, result.uncertain) == (0.0, True)
    _, result = run(monkeypatch, TALL, {90: [0.7], 270: [0.7]})
    assert (result.angle, result.uncertain) == (90.0, True)


def test_the_margin_constants_are_pinned():
    assert orientation.UNCERTAIN_MARGIN_PAIR == 0.13


def test_the_four_way_margin_constant_is_pinned():
    assert orientation.UNCERTAIN_MARGIN_FOUR == 0.41


def test_a_pair_gap_equal_to_the_margin_is_certain_and_below_it_uncertain(monkeypatch):
    gap = orientation.UNCERTAIN_MARGIN_PAIR
    _, at_limit = run(monkeypatch, WIDE, {0: [gap], 180: [0.0]})
    _, below = run(monkeypatch, WIDE, {0: [gap - 0.001], 180: [0.0]})
    assert at_limit.uncertain is False
    assert below.uncertain is True


def test_a_four_way_gap_equal_to_the_margin_is_certain_and_below_it_uncertain(monkeypatch):
    gap = orientation.UNCERTAIN_MARGIN_FOUR
    rest = {90: [0.0] * 7, 180: [0.0] * 7, 270: [0.0] * 7}
    _, at_limit = run(monkeypatch, DIGITS, {0: [gap] * 7, **rest})
    _, below = run(monkeypatch, DIGITS, {0: [gap - 0.001] * 7, **rest})
    assert at_limit.uncertain is False
    assert below.uncertain is True


def test_unusable_scores_do_not_vote(monkeypatch):
    nan = math.nan
    two = np.repeat(WIDE, 2, axis=0)
    _, result = run(monkeypatch, two, {0: [nan, 0.8], 180: [0.5, 0.5]})
    assert (result.angle, result.uncertain) == (0.0, False)
    _, result = run(monkeypatch, two, {0: [nan, nan], 180: [0.5, 0.5]})
    assert result.angle == 180.0  # no usable score counts as 0.0


def test_skew_is_added_to_the_quarter_modulo_a_full_turn(monkeypatch):
    _, result = run(monkeypatch, WIDE, {0: [0.9], 180: [0.2]}, skew=-2)
    assert result.angle == 358.0
    _, result = run(monkeypatch, WIDE, {0: [0.2], 180: [0.9]}, skew=5)
    assert result.angle == 185.0
    _, result = run(monkeypatch, TALL, {90: [0.2], 270: [0.9]}, skew=-100)
    assert result.angle == 170.0


def test_the_detection_is_reused_only_without_skew(monkeypatch):
    doubles, result = run(monkeypatch, WIDE, {0: [0.9], 180: [0.2]})
    assert result.detection is doubles.found[0]
    assert doubles.skewed == [doubles.found[0]]
    doubles, result = run(monkeypatch, WIDE, {0: [0.2], 180: [0.9]})
    assert result.detection is doubles.found[180]
    _, result = run(monkeypatch, WIDE, {0: [0.9], 180: [0.2]}, skew=1)
    assert result.detection is None


def test_the_upright_detection_is_not_repeated(monkeypatch):
    doubles, _ = run(monkeypatch, WIDE, {0: [0.9], 180: [0.2]})
    assert doubles.detected == [0, 180]


def test_only_the_forty_largest_crops_are_read(monkeypatch):
    widths = [0.01 * (i + 1) for i in range(50)]
    boxes = np.array([[0.0, 0.0, w, 0.01] for w in widths])
    scores = {0: [0.5] * 40, 180: [0.5] * 40}
    doubles, _ = run(monkeypatch, boxes, scores)
    for _, crops in doubles.read:
        assert len(crops) == 40
        assert sorted(crops[:, 2]) == pytest.approx(widths[10:])


# --- orientation by Tesseract OSD ------------------------------------------------------------


def with_osd(monkeypatch, first, answer, scores=None, skew=0):
    doubles = Doubles(monkeypatch, first, scores or {}, skew)
    asked = []

    def osd(image):
        asked.append(image)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return doubles, page_orientation(None, IMAGE, osd), asked


def test_an_osd_answer_inside_the_pair_wins_without_any_read(monkeypatch):
    doubles, result, asked = with_osd(monkeypatch, WIDE, Osd(180, 2.0))
    assert (result.angle, result.uncertain) == (180.0, False)
    assert asked == [IMAGE] and doubles.read == [] and doubles.detected == [0, 180]
    assert result.detection is doubles.found[180]
    doubles, result, _ = with_osd(monkeypatch, WIDE, Osd(0, 2.0))
    assert result.angle == 0.0 and doubles.detected == [0] and doubles.read == []
    assert result.detection is doubles.found[0]


def test_the_osd_floor_is_inclusive(monkeypatch):
    floor = orientation.OSD_FLOOR
    scores = {0: [0.2], 180: [0.9]}
    _, used, _ = with_osd(monkeypatch, WIDE, Osd(0, floor), scores)
    assert used.angle == 0.0 and used.uncertain is False
    doubles, voted, _ = with_osd(monkeypatch, WIDE, Osd(0, floor - 0.01), scores)
    assert voted.angle == 180.0 and len(doubles.read) == 2


def test_an_osd_quarter_outside_the_pair_or_no_answer_falls_back_to_the_vote(monkeypatch):
    scores = {0: [0.9], 180: [0.2]}
    for answer in (Osd(90, 5.0), None):
        doubles, result, asked = with_osd(monkeypatch, WIDE, answer, scores)
        assert result.angle == 0.0 and len(doubles.read) == 2 and len(asked) == 1


def test_a_four_way_page_takes_any_osd_quarter(monkeypatch):
    digits = np.array([[0.1 * k, 0.1, 0.1 * k + 0.06, 0.2] for k in range(1, 8)])
    doubles, result, _ = with_osd(monkeypatch, digits, Osd(90, 3.0))
    assert result.angle == 90.0 and doubles.read == []


def test_the_skew_is_added_to_the_osd_quarter(monkeypatch):
    _, result, _ = with_osd(monkeypatch, TALL, Osd(270, 3.0), skew=-2)
    assert result.angle == 268.0 and result.detection is None


def test_the_source_names_who_decided(monkeypatch):
    _, by_osd, _ = with_osd(monkeypatch, WIDE, Osd(180, 2.0))
    assert (by_osd.source, by_osd.error) == ("osd", None)
    scores = {0: [0.9], 180: [0.2]}
    for answer in (None, Osd(0, orientation.OSD_FLOOR - 0.01), Osd(90, 5.0)):
        _, by_vote, _ = with_osd(monkeypatch, WIDE, answer, scores)
        assert (by_vote.source, by_vote.error, by_vote.uncertain) == ("vote", None, False)
    _, plain = run(monkeypatch, WIDE, scores)
    assert plain.source == "vote"


def test_an_osd_failure_votes_and_is_uncertain_whatever_the_margin(monkeypatch):
    scores = {0: [0.9], 180: [0.2]}  # a wide margin: the vote alone would be certain
    failure = TesseractFailure("Tesseract did not report the page orientation within 60 s")
    doubles, result, asked = with_osd(monkeypatch, WIDE, failure, scores)
    assert result.angle == 0.0 and len(doubles.read) == 2 and len(asked) == 1
    assert (result.source, result.uncertain) == ("vote", True)
    assert result.error == str(failure)


def test_a_page_without_words_never_asks_osd(monkeypatch):
    doubles = Doubles(monkeypatch, np.empty((0, 4)), {}, 0)
    asked = []
    result = page_orientation(None, IMAGE, lambda image: asked.append(image))
    assert asked == [] and (result.source, result.error) == ("vote", None)
    assert doubles.read == []


def test_no_osd_function_means_the_vote(monkeypatch):
    doubles, result = run(monkeypatch, WIDE, {0: [0.9], 180: [0.2]})
    assert result.angle == 0.0 and len(doubles.read) == 2


# --- word finding --------------------------------------------------------------------------

PAGE_W, PAGE_H = 1000, 800


def rel(box):
    x0, y0, x1, y1 = box
    return [x0 / PAGE_W, y0 / PAGE_H, x1 / PAGE_W, y1 / PAGE_H]


class Finder:
    """A detector that answers for the whole page or, for a smaller window, a fixed list."""

    def __init__(self, monkeypatch, page_boxes, window_boxes, regions):
        self.windows = []
        self.page_boxes, self.window_boxes = page_boxes, window_boxes
        monkeypatch.setattr(onnxtr_models, "detect", self.detect)
        monkeypatch.setattr("folio_ocr.ocr_words.uncovered_ink", lambda gray, boxes: regions)
        monkeypatch.setattr("folio_ocr.ocr_words.RESCUE_REGIONS", 3)

    def detect(self, models, rgb):
        height, width = rgb.shape[:2]
        if (width, height) == (PAGE_W, PAGE_H):
            self.windows.append("page")
            boxes = self.page_boxes
        else:
            self.windows.append((width, height))
            boxes = [rel_in(b, width, height) for b in self.window_boxes]
        return Detection(np.array(boxes, dtype=np.float64).reshape(-1, 4), MASK)


def rel_in(box, width, height):
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


GRAY = Image.new("L", (PAGE_W, PAGE_H), 255)
RGB = np.asarray(GRAY.convert("RGB"))


def test_only_the_largest_regions_are_searched_again(monkeypatch):
    regions = [(100.0, 100.0, 100.0 + 10 * (i + 1), 110.0) for i in range(20)]
    finder = Finder(monkeypatch, [], [], regions)
    find_words(None, GRAY, RGB, None)
    # window width = region right edge + 64 - (100 - 64); the three widest regions, widest first
    assert finder.windows == ["page", (328, 138), (318, 138), (308, 138)]


def test_a_given_detection_is_not_searched_for_again(monkeypatch):
    regions = [(100.0, 100.0, 120.0, 110.0)]
    finder = Finder(monkeypatch, [], [], regions)
    find_words(None, GRAY, RGB, Detection(np.empty((0, 4)), MASK))
    assert finder.windows == [(148, 138)]


def test_a_second_search_keeps_only_new_words_centred_in_the_region(monkeypatch):
    region = (300.0, 200.0, 340.0, 220.0)  # window: x 236-404, y 136-284 (168 x 148)
    known = (330.0, 200.0, 338.0, 220.0)
    window = [
        (70, 66, 90, 80),  # inside the region, new
        (72, 68, 88, 78),  # inside, but overlaps the word found just before
        (10, 10, 30, 30),  # centre outside the region
        (92, 66, 102, 80),  # inside, but overlaps a known box (x 330-338 is 94-102 here)
    ]
    Finder(monkeypatch, [rel(known)], window, [region])
    boxes = find_words(None, GRAY, RGB, None)
    assert len(boxes) == 2
    assert boxes[0] == pytest.approx(known)
    assert boxes[1] == pytest.approx((306.0, 202.0, 326.0, 216.0))  # back in page coordinates


# --- ink boxes and lines -------------------------------------------------------------------


def ink_page(*rects):
    page = Image.new("L", (200, 100), 255)
    draw = ImageDraw.Draw(page)
    for rect in rects:
        draw.rectangle(rect, fill=0)
    return page


@pytest.fixture(autouse=True)
def margin(monkeypatch):
    monkeypatch.setattr("folio_ocr.ocr_words.LINE_INK_MARGIN", 2.0)


def test_ink_box_shrinks_to_the_ink_inside_the_box():
    page = ink_page((20, 20, 39, 29))
    assert _ink_box(page, (10.0, 10.0, 60.0, 50.0)) == (20.0, 20.0, 40.0, 30.0)


def test_ink_box_reaches_ink_inside_the_margin_but_not_beyond_it():
    assert _ink_box(ink_page((58, 30, 58, 30)), (10.0, 10.0, 57.0, 50.0)) == (
        58.0,
        30.0,
        59.0,
        31.0,
    )
    far = (10.0, 10.0, 57.0, 50.0)
    assert _ink_box(ink_page((60, 30, 60, 30)), far) == far


def test_ink_box_leaves_a_blank_box_unchanged():
    box = (10.5, 10.0, 60.0, 50.0)
    assert _ink_box(ink_page(), box) == box


WORDS = [
    Word((25.0, 0.0, 45.0, 10.0), "b", 0.9),
    Word((0.0, 0.0, 20.0, 10.0), "a", 0.9),
    Word((100.0, 0.0, 120.0, 10.0), "c", 0.9),
]


def texts(lines):
    return [[w.text for w in line.words] for line in lines]


def test_lines_run_left_to_right_for_an_ltr_page():
    assert texts(group_lines(ink_page(), WORDS, "ltr")) == [["a", "b"], ["c"]]


def test_lines_run_right_to_left_for_an_rtl_page_with_words_left_to_right():
    assert texts(group_lines(ink_page(), WORDS, "rtl")) == [["c"], ["a", "b"]]


def test_a_line_box_is_the_ink_of_its_words():
    lines = group_lines(ink_page((5, 3, 38, 7)), WORDS, "ltr")
    assert lines[0].box == (5.0, 3.0, 39.0, 8.0)
    assert lines[1].box == (100.0, 0.0, 120.0, 10.0)  # no ink: the words' own box


def test_a_gap_of_0_2_flags_a_four_way_page_but_not_a_pair_page(monkeypatch):
    scores = {0: [0.9] * 7, 90: [0.0] * 7, 180: [0.7] * 7, 270: [0.0] * 7}
    _, four = run(monkeypatch, DIGITS, scores)
    assert four.angle == 0.0 and four.uncertain is True
    _, pair = run(monkeypatch, WIDE, {0: [0.9], 180: [0.7]})
    assert pair.uncertain is False


def test_branch_and_margin_of_a_pair_page_and_a_four_way_page_by_vote(monkeypatch):
    _, pair = run(monkeypatch, WIDE, {0: [0.9], 180: [0.25]})
    assert (pair.branch, pair.source) == ("pair", "vote")
    assert pair.margin == pytest.approx(0.65)
    _, tall = run(monkeypatch, TALL, {90: [0.5], 270: [0.75]})
    assert (tall.branch, tall.margin) == ("pair", pytest.approx(0.25))
    scores = {0: [0.9] * 7, 90: [0.2] * 7, 180: [0.6] * 7, 270: [0.3] * 7}
    _, four = run(monkeypatch, DIGITS, scores)
    assert four.branch == "four" and four.margin == pytest.approx(0.3)


def test_an_osd_decision_has_the_branch_of_the_ways_and_no_margin(monkeypatch):
    _, pair, _ = with_osd(monkeypatch, WIDE, Osd(180, 2.0))
    assert (pair.source, pair.branch, pair.margin) == ("osd", "pair", None)
    _, four, _ = with_osd(monkeypatch, DIGITS, Osd(90, 3.0))
    assert (four.source, four.branch, four.margin) == ("osd", "four", None)


def test_a_page_with_no_word_has_branch_none_and_no_margin(monkeypatch):
    _, result = run(monkeypatch, np.empty((0, 4)), {})
    assert (result.branch, result.margin, result.source) == ("none", None, "vote")


def test_an_osd_failure_keeps_the_votes_branch_and_margin(monkeypatch):
    failure = TesseractFailure("Tesseract did not report the page orientation within 60 s")
    _, result, _ = with_osd(monkeypatch, WIDE, failure, {0: [0.9], 180: [0.2]})
    assert (result.branch, result.uncertain, result.error) == ("pair", True, str(failure))
    assert result.margin == pytest.approx(0.7)
    scores = {0: [0.9] * 7, 90: [0.2] * 7, 180: [0.6] * 7, 270: [0.3] * 7}
    _, four, _ = with_osd(monkeypatch, DIGITS, failure, scores)
    assert (four.branch, four.uncertain, four.error) == ("four", True, str(failure))
    assert four.margin == pytest.approx(0.3)
