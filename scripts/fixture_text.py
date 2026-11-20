"""Seeded, fictitious document text (English, Arabic and mixed) with labelled entities."""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass

Part = tuple[str, str | None]


@dataclass(frozen=True)
class Entity:
    type: str
    start: int
    end: int


@dataclass(frozen=True)
class TextLine:
    text: str
    direction: str
    entities: tuple[Entity, ...]


@dataclass(frozen=True)
class Row:
    cells: tuple[TextLine, ...]


@dataclass(frozen=True)
class DocText:
    language: str
    layout: str
    direction: str
    rows: tuple[Row, ...]


LANGUAGES = ("en", "ar", "en+ar")
LAYOUTS = ("invoice", "letter")
INDIC_DIGITS = str.maketrans("0123456789,.", "٠١٢٣٤٥٦٧٨٩٬٫")
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

EN = {
    "invoice": "Invoice",
    "inv_no": "Invoice number",
    "date": "Date",
    "bill_to": "Bill to",
    "phone": "Phone",
    "item": "Item",
    "qty": "Qty",
    "unit": "Unit price",
    "amount": "Amount",
    "total": "Total",
    "due": "Due date",
    "prepared": "Prepared by",
    "memo": "Internal memo",
    "to": "To",
    "from": "From",
    "ref": "Ref",
    "subject": "Subject",
    "closing": "Kind regards",
    "subject_text": "Quarterly supplies review",
    "meeting": "The review meeting is set for {}.",
    "budget": "The approved budget is {} for this quarter.",
    "contract": "Contract {} covers the supply of paper and ink.",
    "second": "A second payment of {} is due after delivery.",
    "deposit": "A deposit of {} has already been paid.",
}
AR = {
    "invoice": "فاتورة",
    "inv_no": "رقم الفاتورة",
    "date": "التاريخ",
    "bill_to": "العميل",
    "phone": "الهاتف",
    "item": "البند",
    "qty": "الكمية",
    "unit": "سعر الوحدة",
    "amount": "المبلغ",
    "total": "الإجمالي",
    "due": "تاريخ الاستحقاق",
    "prepared": "أعدها",
    "memo": "مذكرة داخلية",
    "to": "إلى",
    "from": "من",
    "ref": "المرجع",
    "subject": "الموضوع",
    "closing": "مع خالص التحية",
    "subject_text": "مراجعة التوريدات الفصلية",
    "meeting": "نود إبلاغكم بأن اجتماع المراجعة سيعقد بتاريخ {}",
    "budget": "الميزانية المعتمدة لهذا الربع هي {}",
    "contract": "العقد رقم {} يغطي توريد الورق والأقلام",
    "second": "الدفعة الثانية بقيمة {} تستحق بعد التسليم",
    "deposit": "تم استلام دفعة مقدمة بقيمة {}",
}
EN_NAMES = (
    "Daniel Mercer",
    "Priya Natarajan",
    "Owen Calloway",
    "Hannah Brightwell",
    "Marcus Lindqvist",
    "Fiona Achebe",
    "Tomas Okafor",
    "Grace Whitlock",
)
AR_NAMES = (
    "أحمد الحسن",
    "سارة العتيبي",
    "خالد المنصوري",
    "ليلى الشمري",
    "يوسف القحطاني",
    "منى الزهراني",
    "عمر الدوسري",
    "نورة السبيعي",
)
EN_COMPANIES = ("Northwind Office Supply Ltd", "Harborview Trading Co.")
AR_COMPANIES = ("شركة الأفق للتجارة", "مؤسسة النخبة للتوريدات")
EN_ITEMS = (
    "Ballpoint pens (box of 50)",
    "A4 paper ream",
    "Desk stapler",
    "Archive folders",
    "Printer toner",
)
AR_ITEMS = ("أقلام حبر", "ورق طباعة", "دباسة مكتبية", "ملفات حفظ", "حبر طابعة")


def line_direction(text: str) -> str:
    """Direction of the first strong character: 'rtl' for R/AL, else 'ltr'."""
    for ch in text:
        bidi = unicodedata.bidirectional(ch)
        if bidi in ("L", "R", "AL"):
            return "ltr" if bidi == "L" else "rtl"
    return "ltr"


def build_line(parts: list[Part]) -> TextLine:
    """Join (text, entity_type_or_None) parts; entity spans are computed from the parts."""
    text, entities = "", []
    for piece, kind in parts:
        if kind:
            entities.append(Entity(kind, len(text), len(text) + len(piece)))
        text += piece
    if not unicodedata.is_normalized("NFC", text):
        raise ValueError(f"line is not NFC: {text!r}")
    return TextLine(text, line_direction(text), tuple(entities))


def _row(*cells: list[Part]) -> Row:
    return Row(tuple(build_line(c) for c in cells))


class _Values:
    """Draws names, dates, amounts, ids and phones in the style of one language."""

    def __init__(self, language: str, rng: random.Random):
        self.language, self.rng = language, rng
        self.latin = language != "ar"
        pool = AR_NAMES if language == "ar" else EN_NAMES
        self._names = rng.sample(pool, len(pool))

    def name(self) -> str:
        return self._names.pop()

    def company(self) -> str:
        return self.rng.choice(EN_COMPANIES if self.language == "en" else AR_COMPANIES)

    def items(self, count: int) -> list[str]:
        return self.rng.sample(EN_ITEMS if self.language == "en" else AR_ITEMS, count)

    def date(self) -> str:
        day, month = self.rng.randint(1, 28), self.rng.randint(1, 12)
        if self.language == "en":
            if self.rng.random() < 0.5:
                return f"{day} {MONTHS[month - 1]} 2026"
            return f"2026-{month:02d}-{day:02d}"
        if self.rng.random() < 0.5:
            return f"{day:02d}/{month:02d}/2026".translate(INDIC_DIGITS)
        return f"2026/{month:02d}/{day:02d}"

    def cents(self, low: int, high: int) -> int:
        return self.rng.randint(low, high) * 25

    def money(self, cents: int) -> str:
        grouped = f"{cents // 100:,}.{cents % 100:02d}"
        if self.language == "en":
            return grouped
        if self.rng.random() < 0.5:
            return grouped.translate(INDIC_DIGITS)
        return grouped.replace(",", "")

    def ident(self, prefix: str) -> str:
        number = self.rng.randint(1000, 9999)
        return f"{prefix}-2026-{number}" if self.latin else f"2026{number}"

    def phone(self) -> str:
        if self.language == "en":
            return f"+44 20 7946 {self.rng.randint(1000, 9999)}"
        if self.language == "en+ar":
            return f"+966 50 {self.rng.randint(100, 999)} {self.rng.randint(1000, 9999)}"
        digits = f"05{self.rng.randint(10000000, 99999999)}"
        return digits.translate(INDIC_DIGITS) if self.rng.random() < 0.5 else digits


def _labelled(label: str, value: str, kind: str | None) -> Row:
    return _row([(f"{label}: ", None), (value, kind)])


def _phone_row(v: _Values, labels: dict) -> Row:
    label = "Phone" if v.language == "en+ar" else labels["phone"]
    return _labelled(label, v.phone(), "phone")


def _invoice(v: _Values, t: dict) -> list[Row]:
    rows = [
        _row([(f"{t['invoice']}: {v.company()}", None)]),
        _labelled(t["inv_no"], v.ident("INV"), "id"),
        _labelled(t["date"], v.date(), "date"),
        _labelled(t["bill_to"], v.name(), "name"),
        _phone_row(v, t),
        _row(*[[(t[k], None)] for k in ("item", "qty", "unit", "amount")]),
    ]
    total = 0
    for item in v.items(3):
        unit, qty = v.cents(8, 960), v.rng.randint(1, 3)
        total += unit * qty
        qty_text = str(qty)
        if v.language != "en" and v.rng.random() < 0.5:
            qty_text = qty_text.translate(INDIC_DIGITS)
        rows.append(
            _row(
                [(item, None)],
                [(qty_text, None)],
                [(v.money(unit), "amount")],
                [(v.money(unit * qty), "amount")],
            )
        )
    rows += [
        _labelled(t["total"], v.money(total), "amount"),
        _labelled(t["due"], v.date(), "date"),
        _labelled(t["prepared"], v.name(), "name"),
    ]
    return rows


def _letter(v: _Values, t: dict) -> list[Row]:
    big = v.cents(400, 4000)
    body = [
        _labelled(t["subject"], t["subject_text"], None),
        _sentence(t["meeting"], v.date(), "date"),
        _sentence(t["budget"], v.money(big), "amount"),
        _sentence(t["contract"], v.ident("CT"), "id"),
        _sentence(t["second"], v.money(big // 2), "amount"),
        _sentence(t["deposit"], v.money(big // 4), "amount"),
    ]
    head = [
        _row([(t["memo"], None)]),
        _labelled(t["to"], v.name(), "name"),
        _labelled(t["from"], v.name(), "name"),
        _labelled(t["date"], v.date(), "date"),
        _labelled(t["ref"], v.ident("REF"), "id"),
    ]
    tail = [_phone_row(v, t), _row([(t["closing"], None)]), _row([(v.name(), "name")])]
    return head + body + tail


def _sentence(template: str, value: str, kind: str) -> Row:
    before, after = template.split("{}")
    parts: list[Part] = [(before, None), (value, kind)]
    if after:
        parts.append((after, None))
    return _row(parts)


def make_document(language: str, layout: str, rng: random.Random) -> DocText:
    """Build one seeded document; the same rng state always gives the same document."""
    if language not in LANGUAGES:
        raise ValueError(f"unknown language: {language!r}")
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout: {layout!r}")
    values = _Values(language, rng)
    labels = EN if language == "en" else AR
    rows = _invoice(values, labels) if layout == "invoice" else _letter(values, labels)
    return DocText(language, layout, "ltr" if language == "en" else "rtl", tuple(rows))


# Plain, entity-free lines for the pages that stress number, date, punctuation and mixed-script handling.
AR_ADVERSARIAL = (
    "المبلغ 350.75 ريال",
    "القيمة ٣٥٠٫٧٥ ريال",
    "العدد ١٬٢٥٠ قطعة",
    "المجموع 1,250.75 ريال",
    "الرقم ۴۵ فقط",
    "التاريخ ١٢/٠٥/٢٠٢٦",
    "التاريخ 12/05/2026",
    "الموعد ٢٠٢٦-٠٥-١٢",
    "الموعد 2026-05-12",
    "العقد رقم CT-2026-3587 ساري",
    "الهاتف +966 50 123 4567",
    "المبلغ 1 250 000 ريال",
    "المبلغ ١ ٢٥٠ ٠٠٠ ريال",
    "50% من المبلغ",
    "خصم 50% على الطلب",
    "٥٠٪ من المبلغ",
    "خصم ٥٠٪ على الطلب",
    "انظر (مثال) أدناه",
    "البند (٣) من العقد",
    "السيد (Tomas) حاضر",
    "الاسم: Tomas Okafor",
    "البريد: tomas.okafor@example.com للتواصل",
    "هل تم الدفع؟ نعم؛ تم، شكرا",
    "السلام عليكم ورحمة الله",
    "شُكْرًا جَزِيلًا لَكُمْ",
    "مـــرحبا بالجميع",
    "Meeting with أحمد الحسن on Monday",
    "350 boxes were delivered on time",
)
EN_ADVERSARIAL = (
    "The invoice total is 350.75 for this quarter.",
    "Payment of 1,250.75 was received on 12 May 2026.",
    "تم إرسال الفاتورة إلى Northwind Office Supply",
    "Please call +44 20 7946 1234 before Friday.",
    "The contract CT-2026-3587 covers paper and ink.",
    "Kind regards",
)
# Word-style bulleted pages: heading, paragraph, 3 bullets, paragraph, 2 bullets, closing line.
EN_BULLETED = (
    "Office supplies update",
    "The following items were checked at the Harborview store this week.",
    "Ballpoint pens are in stock",
    "A4 paper needs reordering",
    "Printer toner arrives on Friday",
    "Two actions remain before the end of the month.",
    "Confirm the new paper supplier",
    "Archive the old delivery notes",
    "Thank you for your help",
)
AR_BULLETED = (
    "تحديث المستلزمات المكتبية",
    "تمت مراجعة البنود التالية في المخزن هذا الأسبوع.",
    "الأقلام متوفرة في المخزن",
    "ورق الطباعة يحتاج إلى إعادة طلب",
    "حبر الطابعة يصل يوم الخميس",
    "بقي إجراءان قبل نهاية الشهر.",
    "تأكيد مورد الورق الجديد",
    "حفظ إشعارات التسليم القديمة",
    "شكرا لتعاونكم",
)
BULLET_ROWS = ("text", "text", "bullet", "bullet", "bullet", "text", "bullet", "bullet", "text")
