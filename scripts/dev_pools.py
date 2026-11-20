"""Fictitious vocabulary for the development-variant documents: labels, names, items, sentences."""

from __future__ import annotations

from collections.abc import Mapping

EN_LABELS = {
    "printed": "Printed at",
    "subtotal": "Subtotal",
    "vat": "VAT",
    "due": "Amount due",
    "paid": "Paid by",
    "bye": "Thank you, see you soon",
    "actions": "Action items",
    "name": "Full name",
    "born": "Born on",
    "mobile": "Mobile",
    "email": "E-mail",
    "member": "Member no.",
    "issued": "Issued on",
    "balance": "Balance",
    "city": "City",
    "employer": "Employer",
    "street": "Street",
}
AR_LABELS = {
    "printed": "تاريخ الطباعة",
    "subtotal": "المجموع الفرعي",
    "vat": "ضريبة القيمة المضافة",
    "due": "المستحق",
    "paid": "طريقة الدفع",
    "bye": "شكرا لزيارتكم ونراكم قريبا",
    "actions": "بنود المتابعة",
    "name": "الاسم الكامل",
    "born": "تاريخ الميلاد",
    "mobile": "الجوال",
    "email": "البريد الإلكتروني",
    "member": "رقم العضوية",
    "issued": "تاريخ الإصدار",
    "balance": "الرصيد",
    "city": "المدينة",
    "employer": "جهة العمل",
    "street": "العنوان",
}
EN_COLUMNS = ("SKU", "Batch", "Units", "Cost", "Ship date")
AR_COLUMNS = ("الرمز", "الدفعة", "الوحدات", "التكلفة", "تاريخ الشحن")
EN_NAMES = (
    "Elena Rossetti", "Kofi Mensah", "Ingrid Halvorsen", "Rafael Duarte", "Mei Tanaka",
    "Olumide Adeyemi", "Sofia Karagianni", "Viktor Novak", "Amara Singh", "Lucas Beaumont",
    "Zainab Farouk", "Henrik Dahl", "Noor Haddad",
)  # fmt: skip
AR_NAMES = (
    "فاطمة الجابري", "إبراهيم الخطيب", "هدى الكبيسي", "طارق العمري", "رنا البيطار",
    "ماجد الحربي", "دعاء الصالحي", "سلمان الغامدي", "آمنة الرشيد", "بشير الأنصاري",
    "ريم الخالدي", "حمزة البغدادي",
)  # fmt: skip
EN_ORGS = (
    "Bluepeak Logistics GmbH", "Cedar & Finch Stationers", "Meridian Cold Storage SA",
    "Tidewater Foods Ltd", "Pinecrest Hardware", "Alder Row Bakery",
)  # fmt: skip
AR_ORGS = (
    "مخبز واحة النخيل", "شركة دجلة للمقاولات", "مؤسسة الرافدين للأغذية",
    "متجر بيت الزيتون", "شركة سهول الشام للنقل", "مكتبة الحكمة",
)  # fmt: skip
EN_ITEMS = (
    "Olive oil 750 ml", "Basmati rice 5 kg", "Green tea (20 bags)", "Notebook A5",
    "Hand soap refill", "LED bulb 9W", "Espresso beans 1 kg", "Paper napkins",
    "Dish sponge pack", "Rechargeable batteries", "Wool socks", "Greek yoghurt 500 g",
    "Sunflower seeds",
)  # fmt: skip
AR_ITEMS = (
    "زيت زيتون", "أرز بسمتي", "شاي أخضر", "دفتر ملاحظات", "صابون سائل", "مصباح موفر",
    "قهوة مطحونة", "مناديل ورقية", "إسفنج جلي", "بطاريات قابلة للشحن", "جوارب صوفية",
    "لبن زبادي", "بذور دوار الشمس", "منظف زجاج",
)  # fmt: skip
EN_STREETS = ("Willow Lane", "Quay Street", "Orchard Road", "Mill Court", "Harbour Walk")
AR_STREETS = ("شارع الورود", "شارع الجامعة", "طريق الميناء", "شارع المتنبي")
EN_CITIES = ("Leeds", "Gdansk", "Porto", "Tallinn", "Graz")
AR_CITIES = ("الرباط", "مسقط", "عمان", "تونس", "بيروت")
EN_PAYMENTS = ("card", "cash", "mobile wallet")
AR_PAYMENTS = ("بطاقة", "نقدا", "محفظة إلكترونية")
EN_TITLES = (
    "Site inspection summary", "Warehouse stock note", "Facilities maintenance log",
    "Supplier onboarding brief",
)  # fmt: skip
AR_TITLES = ("ملخص معاينة الموقع", "مذكرة جرد المستودع", "سجل صيانة المرافق", "موجز تأهيل الموردين")
EN_ACTIONS = (
    "Confirm the delivery window", "Update the stock sheet", "Send the signed form back",
    "Book the cold room", "Check the supplier invoice", "Archive the old batch records",
)  # fmt: skip
AR_ACTIONS = (
    "تأكيد موعد التسليم", "تحديث جدول المخزون", "إعادة النموذج الموقع", "حجز غرفة التبريد",
    "مراجعة فاتورة المورد", "أرشفة سجلات الدفعات القديمة",
)  # fmt: skip
CURRENCIES_EN = ("CHF", "SEK", "PLN", "NOK")
CURRENCIES_AR = ("ر.ع", "د.ك", "د.ت", "ر.ق")
MONTHS_ABBR = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)  # fmt: skip
EMAIL_DOMAINS = ("mailbox.example", "post.example", "inbox.example")
ID_PREFIXES = ("TX", "RC", "WB", "KD")
SKU_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
GRID_LATIN = "ABCDEFGHJKLMNPRSTUVWXYZabdefghjkmnqrty"
GRID_ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
SYMBOLS = "0123456789/.:,@+-%&()"

# (template, entity kind of the value slot)
EN_SENTENCES = (
    ("The delivery arrived at the depot on {}.", "date"),
    ("The outstanding balance stands at {} including tax.", "amount"),
    ("Please quote order {} in all correspondence.", "id"),
    ("The file will be followed up by {} next week.", "name"),
    ("The committee will meet on {} at the head office.", "date"),
    ("A sum of {} was deducted from the current account.", "amount"),
    ("Support can be reached on {} during working days.", "phone"),
    ("The board approved a financial allocation of {}.", "amount"),
    ("The final draft of contract {} went to the legal adviser.", "id"),
    ("The subscription expired on {} and must be renewed.", "date"),
    ("Test results show that batch {} meets the specification.", "id"),
    ("The annual summary was prepared by {} with great care.", "name"),
    ("The estimated cost of the maintenance work is {}.", "amount"),
    ("The supplier was represented by {} and several staff.", "name"),
)
AR_SENTENCES = (
    ("تم تسليم الشحنة إلى المستودع بتاريخ {}", "date"),
    ("بلغت قيمة الفاتورة المستحقة {} شاملة الضريبة", "amount"),
    ("يرجى الإشارة إلى رقم الطلب {} عند المراسلة", "id"),
    ("سيتولى متابعة الملف المسؤول {} خلال الأسبوع القادم", "name"),
    ("تقرر عقد اجتماع اللجنة يوم {} في مقر الشركة", "date"),
    ("تم خصم مبلغ {} من الحساب الجاري", "amount"),
    ("يمكن الاتصال بقسم الدعم على الرقم {} طوال أيام العمل", "phone"),
    ("وافق مجلس الإدارة على الاعتماد المالي بقيمة {}", "amount"),
    ("أرسلت المسودة النهائية للعقد رقم {} إلى المراجع القانوني", "id"),
    ("انتهت صلاحية الاشتراك بتاريخ {} ويلزم تجديدها", "date"),
    ("تشير نتائج الفحص إلى أن الدفعة {} مطابقة للمواصفات", "id"),
    ("أعد التقرير السنوي الأستاذ {} بعناية كبيرة", "name"),
    ("تبلغ الكلفة التقديرية لأعمال الصيانة {} للموقع الرئيسي", "amount"),
    ("حضر الاجتماع ممثل المورد {} وعدد من الموظفين", "name"),
)

POOLS: Mapping[str, tuple[str, ...]] = {
    "en_labels": tuple(EN_LABELS.values()),
    "ar_labels": tuple(AR_LABELS.values()),
    "en_columns": EN_COLUMNS,
    "ar_columns": AR_COLUMNS,
    "en_names": EN_NAMES,
    "ar_names": AR_NAMES,
    "en_orgs": EN_ORGS,
    "ar_orgs": AR_ORGS,
    "en_items": EN_ITEMS,
    "ar_items": AR_ITEMS,
    "en_streets": EN_STREETS,
    "ar_streets": AR_STREETS,
    "en_cities": EN_CITIES,
    "ar_cities": AR_CITIES,
    "en_payments": EN_PAYMENTS,
    "ar_payments": AR_PAYMENTS,
    "en_titles": EN_TITLES,
    "ar_titles": AR_TITLES,
    "en_actions": EN_ACTIONS,
    "ar_actions": AR_ACTIONS,
    "en_currencies": CURRENCIES_EN,
    "ar_currencies": CURRENCIES_AR,
    "en_months": MONTHS_ABBR,
    "en_domains": EMAIL_DOMAINS,
    "en_prefixes": ID_PREFIXES,
    "en_sentences": tuple(t for t, _ in EN_SENTENCES),
    "ar_sentences": tuple(t for t, _ in AR_SENTENCES),
    "en_symbols": (SYMBOLS, SKU_LETTERS, GRID_LATIN),
    "ar_digits": (INDIC_DIGITS, GRID_ARABIC),
}
