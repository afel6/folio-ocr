"""Reading the development set for calibration: documents in manifest order per cell, stopping
a cell once it holds enough bad lines or reaches the document cap."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import accuracy_reads as ar
import accuracy_stats as stats
import heldout


@dataclass(frozen=True)
class DevCell:
    units: tuple[stats.Unit, ...] = ()
    tally: heldout.Tally = field(default_factory=heldout.Tally)
    seconds: tuple[float, ...] = ()
    stopped_by: str | None = None


def stop_reason(bad_lines: int, docs_with_bad: int, documents: int) -> str | None:
    """Why a cell takes no more documents, or None while it should keep going."""
    if bad_lines >= stats.DEV_MIN_BAD and docs_with_bad >= stats.DEV_MIN_DOCS:
        return "minimums"
    if documents >= stats.DEV_CAP_DOCS:
        return "cap"
    return None


def _absorb(cell: DevCell, doc: dict, read: Any) -> DevCell:
    unit = stats.Unit(doc["unit"], (read.outcome,))
    tally = heldout.add(cell.tally, stats.unit_counts(unit, 0.0)[0])
    units = (*cell.units, unit)
    return replace(
        cell,
        units=units,
        tally=tally,
        seconds=(*cell.seconds, read.seconds),
        stopped_by=stop_reason(tally.bad_lines, tally.documents_with_bad, len(units)),
    )


def collect_dev(
    docs: Sequence[dict], read: Callable[[tuple], Any], jobs: int, plan: ar.CellPlan = ar.DEFAULT_PLAN
) -> dict[str, DevCell]:
    """Per cell, the documents up to its stop point. `read(task)` reads one (doc, cell, mode,
    with_tesseract) task; up to `jobs` documents are read ahead, but those past a cell's stop are dropped."""
    cells = {name: DevCell() for name in plan.cells}
    for start in range(0, len(docs), jobs):
        tasks = [
            (d, cell, mode, with_tesseract)
            for d in docs[start : start + jobs]
            for cell, mode, with_tesseract in ar.cell_reads(d["language"], plan)
            if cells[cell].stopped_by is None
        ]
        for (doc, cell, _mode, _with), got in zip(tasks, ar.parallel_map(read, tasks, jobs), strict=True):
            if cells[cell].stopped_by is None:
                cells[cell] = _absorb(cells[cell], doc, got)
    return {
        name: c if c.stopped_by else replace(c, stopped_by="dev set end") for name, c in cells.items()
    }
