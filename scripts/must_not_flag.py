"""Correct text lines the order check must not flag, and lines that are only measured.

RTL lines are shown in right-to-left paragraphs, LTR lines in left-to-right ones. Every
string is NFC.
"""

from __future__ import annotations

import unicodedata


def _nfc(lines: list[str]) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", line) for line in lines)


RTL = _nfc(
    [
        "المبلغ 350.75 ريال",
        "القيمة ٣٥٠٫٧٥ ريال",
        "العدد ١٬٢٥٠ قطعة",
        "المجموع 1,250.75 ريال",
        "الرقم ۴۵ فقط",
        "التاريخ ١٢/٠٥/٢٠٢٦",
        "التاريخ 12/05/2026",
        "الموعد ٢٠٢٦-٠٥-١٢",
        "العقد رقم CT-2026-3587 ساري",
        "+966 50 123 4567 للاتصال",
        "الهاتف +966 50 123 4567",
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
        "نعم/لا و/أو",
        'قال "نعم" للعرض',
        "قال «نعم» للعرض",
        "انتظر... ثم تابع",
        "انتظر… ثم تابع",
        "التسليم - الفوترة – الدفع — الإغلاق",
        "الفريق Smith - Jones – Brown — Lee",
        "١) البند الأول",
        "١) يُدفع المبلغ (خلال",
        "المادة(٣)من العقد",
        "السيد(Tomas)حاضر",
    ]
)

LTR = _nfc(
    [
        "1) Pay the invoice (by",
        "a) first item",
        "see (page",
        "Total: 350.75",
        "+966 50 123 4567 for calls",
        "Call +966 50 123 4567 today",
        "50% of the amount",
        "yes/no and/or 12/05/2026",
        'He said "yes" to the offer',
        "He said «yes» to the offer",
        "Wait... then continue",
        "Wait… then continue",
        "pay - send – file — close",
        "Name: أحمد الحسن",
    ]
)

# Measured and reported, never gated: (text, paragraph direction)
MEASURE_ONLY = (
    ("Friday) and send (copy", "ltr"),
    (unicodedata.normalize("NFC", "الجمعة) ثم أرسل (نسخة"), "rtl"),
    (unicodedata.normalize("NFC", "Meeting with أحمد الحسن on Monday"), "rtl"),
    (unicodedata.normalize("NFC", "تم إرسال الفاتورة إلى Northwind Office Supply"), "ltr"),
    ("350 boxes were delivered on time", "rtl"),
)
