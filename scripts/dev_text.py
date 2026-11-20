"""Seeded text for the development-variant documents: receipt, report, table, form and grid."""

from __future__ import annotations

import random

import dev_pools as p
from dev_pools import POOLS  # noqa: F401  (re-exported for the pool checks)
from fixture_text import DocText, Part, Row, build_line

LANGUAGES = ("en", "ar", "en+ar")
TEMPLATES = ("receipt", "report", "table", "form", "grid")
INDIC = str.maketrans("0123456789", p.INDIC_DIGITS)


def _row(*cells: list[Part]) -> Row:
    return Row(tuple(build_line(c) for c in cells))


def _labelled(label: str, value: str, kind: str | None) -> list[Part]:
    return [(f"{label}: ", None), (value, kind)]


class _Values:
    """Draws names, dates, amounts, ids and phones in the style of one language.

    Arabic and mixed documents use Arabic labels and amounts; mixed ones keep Latin names, codes,
    e-mails and phones. Each document writes its Arabic digits either as ASCII or as Arabic-Indic.
    """

    def __init__(self, language: str, rng: random.Random):
        self.rng = rng
        self.arabic_labels = language != "en"
        self.latin_values = language != "ar"
        self.labels = p.AR_LABELS if self.arabic_labels else p.EN_LABELS
        self.indic = self.arabic_labels and rng.random() < 0.5
        self.european = rng.random() < 0.5
        self.currency = rng.choice(p.CURRENCIES_AR if self.arabic_labels else p.CURRENCIES_EN)
        self.currency_first = rng.random() < 0.5

    def digits(self, text: str) -> str:
        return text.translate(INDIC) if self.indic else text

    def pick(self, en: tuple[str, ...], ar: tuple[str, ...], latin: bool | None = None) -> str:
        latin = self.latin_values if latin is None else latin
        return self.rng.choice(en if latin else ar)

    def name(self) -> str:
        return self.pick(p.EN_NAMES, p.AR_NAMES)

    def org(self) -> str:
        return self.pick(p.EN_ORGS, p.AR_ORGS)

    def city(self) -> str:
        return self.pick(p.EN_CITIES, p.AR_CITIES)

    def address(self) -> str:
        number = str(self.rng.randint(2, 98))
        street = self.pick(p.EN_STREETS, p.AR_STREETS)
        if self.latin_values:
            return f"{number} {street}, {self.city()}"
        return f"{street} {self.digits(number)} - {self.city()}"

    def date(self) -> str:
        day, month = self.rng.randint(1, 28), self.rng.randint(1, 12)
        year = self.rng.randint(2027, 2029)
        formats = [f"{year}.{month:02d}.{day:02d}", f"{day:02d}/{month:02d}/{year}"]
        if not self.arabic_labels:
            formats.append(f"{day:02d} {p.MONTHS_ABBR[month - 1]} {year}")
        return self.digits(self.rng.choice(formats))

    def time(self) -> str:
        return self.digits(f"{self.rng.randint(7, 21):02d}:{self.rng.randint(0, 59):02d}")

    def cents(self, low: int, high: int) -> int:
        return self.rng.randint(low, high) * 5

    def money(self, cents: int) -> str:
        whole, frac = divmod(cents, 100)
        if self.arabic_labels:
            text = f"{whole}.{frac:02d}"
        elif self.european:
            text = f"{whole:,}".replace(",", ".") + f",{frac:02d}"
        else:
            text = f"{whole:,}.{frac:02d}"
        return self.digits(text)

    def amount(self, cents: int) -> str:
        text = self.money(cents)
        return f"{self.currency} {text}" if self.currency_first else f"{text} {self.currency}"

    def qty(self) -> str:
        return self.digits(str(self.rng.randint(1, 240)))

    def ident(self) -> str:
        if self.latin_values:
            # Long ids, wider than 8:1 in most fonts, so dev pages hold words the recogniser
            # reads in parts.
            prefix, year = self.rng.choice(p.ID_PREFIXES), self.rng.randint(2027, 2029)
            branch, number = self.rng.randint(100, 999), self.rng.randint(1, 999999)
            return f"{prefix}/{year}/{branch}/{number:06d}"
        return self.digits(str(self.rng.randint(1000000, 9999999)))

    def sku(self) -> str:
        if not self.latin_values:
            return self.ident()
        letters = "".join(self.rng.choice(p.SKU_LETTERS) for _ in range(2))
        if self.rng.random() < 0.5:
            return f"{letters}-{self.rng.randint(100, 999)}"
        lot, item = self.rng.randint(10000, 99999), self.rng.randint(100, 999)
        return f"{letters}{self.rng.randint(100, 999)}-{lot}-{item}"

    def phone(self) -> str:
        tail = f"{self.rng.randint(1000, 9999)} {self.rng.randint(1000, 9999)}"
        text = f"+{self.rng.choice((968, 212, 420))} {self.rng.randint(1, 9)}{tail}"
        return text if self.latin_values else self.digits(text)

    def email(self) -> str:
        return self.name().lower().replace(" ", ".") + "@" + self.rng.choice(p.EMAIL_DOMAINS)

    def value(self, kind: str) -> str:
        if kind == "amount":
            return self.amount(self.cents(400, 90000))
        return {"date": self.date, "name": self.name, "id": self.ident, "phone": self.phone}[kind]()


def _sentence(template: str, value: str, kind: str) -> list[Part]:
    before, after = template.split("{}")
    parts: list[Part] = [(before, None), (value, kind)]
    return parts + [(after, None)] if after else parts


def _receipt(v: _Values) -> list[Row]:
    lab = v.labels
    rows = [
        _row([(v.org(), None)]),
        _row([(v.address(), None)]),
        _row([*_labelled(lab["printed"], v.date(), "date"), (" " + v.time(), None)]),
    ]
    total = 0
    items = p.AR_ITEMS if v.arabic_labels else p.EN_ITEMS
    for item in v.rng.sample(items, v.rng.randint(4, 7)):
        price = v.cents(20, 1800)
        total += price
        parts: list[Part] = [(item, None)]
        if v.arabic_labels and v.latin_values:
            parts.append((" " + v.sku(), "id"))
        rows.append(_row(parts, [(v.money(price), "amount")]))
    vat = total * v.rng.choice((5, 8, 19, 20)) // 100
    for key, cents, show in (
        ("subtotal", total, v.money),
        ("vat", vat, v.money),
        ("due", total + vat, v.amount),
    ):
        rows.append(_row([(lab[key], None)], [(show(cents), "amount")]))
    pay = v.pick(p.EN_PAYMENTS, p.AR_PAYMENTS, not v.arabic_labels)
    return [*rows, _row(_labelled(lab["paid"], pay, None)), _row([(lab["bye"], None)])]


def _report(v: _Values) -> list[Row]:
    ar = v.arabic_labels
    pool, actions = (p.AR_SENTENCES, p.AR_ACTIONS) if ar else (p.EN_SENTENCES, p.EN_ACTIONS)
    rows = [_row([(v.rng.choice(p.AR_TITLES if ar else p.EN_TITLES), None)])]
    for template, kind in v.rng.sample(pool, v.rng.randint(6, 10)):
        rows.append(_row(_sentence(template, v.value(kind), kind)))
    rows.append(_row([(v.labels["actions"], None)]))
    for i, action in enumerate(v.rng.sample(actions, v.rng.randint(3, 5)), 1):
        rows.append(_row([(f"{v.digits(str(i))}. {action}", None)]))
    return rows


def _table(v: _Values) -> list[Row]:
    columns = p.AR_COLUMNS if v.arabic_labels else p.EN_COLUMNS
    picked = v.rng.choice(((0, 2, 3, 4), (0, 1, 2, 3, 4)))
    makers = {
        0: lambda: [(v.sku(), "id")],
        1: lambda: [(v.ident(), "id")],
        2: lambda: [(v.qty(), None)],
        3: lambda: [(v.money(v.cents(100, 250000)), "amount")],
        4: lambda: [(v.date(), "date")],
    }
    rows = [_row(*[[(columns[i], None)] for i in picked])]
    for _ in range(v.rng.randint(6, 10)):
        rows.append(_row(*[makers[i]() for i in picked]))
    return rows


def _form(v: _Values) -> list[Row]:
    fields = [
        ("name", v.name, "name"),
        ("born", v.date, "date"),
        ("mobile", v.phone, "phone"),
        ("member", v.ident, "id"),
        ("issued", v.date, "date"),
        ("balance", lambda: v.amount(v.cents(400, 90000)), "amount"),
        ("city", v.city, None),
        ("employer", v.org, None),
        ("street", v.address, None),
    ]
    if v.latin_values:
        fields.insert(3, ("email", v.email, None))
    chosen = v.rng.sample(fields, v.rng.randint(6, min(10, len(fields))))
    cells = [_labelled(v.labels[key], make(), kind) for key, make, kind in chosen]
    return [_row(*cells[i : i + 2]) for i in range(0, len(cells), 2)]


def _grid(v: _Values) -> list[Row]:
    scripts = []
    if v.latin_values:
        scripts.append(p.GRID_LATIN)
    if v.arabic_labels:
        scripts.append(p.GRID_ARABIC)
    alphabet = v.rng.choice((v.digits("0123456789"), v.rng.choice(scripts)))
    cols = v.rng.randint(6, 10)
    return [
        _row(*[[(v.rng.choice(alphabet), None)] for _ in range(cols)])
        for _ in range(v.rng.randint(8, 12))
    ]


_BUILDERS = {"receipt": _receipt, "report": _report, "table": _table, "form": _form, "grid": _grid}


def make_dev_document(language: str, template: str, rng: random.Random) -> DocText:
    """Build one seeded document; the same rng state always gives the same document."""
    if language not in LANGUAGES:
        raise ValueError(f"unknown language: {language!r}")
    if template not in TEMPLATES:
        raise ValueError(f"unknown template: {template!r}")
    rows = _BUILDERS[template](_Values(language, rng))
    return DocText(language, template, "ltr" if language == "en" else "rtl", tuple(rows))
