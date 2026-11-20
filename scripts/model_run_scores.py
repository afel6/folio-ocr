"""Checks over the per-line records: which score ranks good lines over bad, and which lines a
threshold could be pinned to."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import accuracy_eval

STOP_MARGIN = 0.02
MAX_EXAMPLES = 50


def auc(good: Sequence[float], bad: Sequence[float]) -> float | None:
    """Probability that a random good score exceeds a random bad one; ties count half."""
    if not good or not bad:
        return None
    ranked = sorted([(s, True) for s in good] + [(s, False) for s in bad])
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(1 for _, is_good in ranked[i:j] if is_good)
        i = j
    return (rank_sum - len(good) * (len(good) + 1) / 2) / (len(good) * len(bad))


def _logprobs(rec: dict) -> list[float] | None:
    entries = rec.get("entries")
    if not isinstance(entries, list) or not entries:
        return None
    values = []
    for e in entries:
        v = e.get("logprob") if isinstance(e, dict) else None
        if isinstance(v, bool) or not isinstance(v, int | float) or math.isnan(v) or v == math.inf:
            return None
        values.append(float(v))
    return values


def _exp(x: float) -> float:
    return math.exp(min(x, 700.0))


def aggregation_check(records: Sequence[dict]) -> dict:
    """How well exp(min), exp(mean) and exp(sum) of a line's logprobs rank good lines over bad."""
    scores: dict[str, tuple[list[float], list[float]]] = {
        k: ([], []) for k in ("exp_min", "exp_mean", "exp_sum")
    }
    for rec in records:
        values = _logprobs(rec) if rec["reader"] == "model" and rec["matched"] else None
        if values is None or rec["cer"] is None:
            continue
        side = 0 if rec["cer"] <= accuracy_eval.BAD_CER else 1
        scores["exp_min"][side].append(_exp(min(values)))
        scores["exp_mean"][side].append(_exp(sum(values) / len(values)))
        scores["exp_sum"][side].append(_exp(sum(values)))
    result = {k: auc(*pair) for k, pair in scores.items()}
    others = [result["exp_mean"], result["exp_sum"]]
    complete = result["exp_min"] is not None and None not in others
    margin = max(others) - result["exp_min"] if complete else None
    good, bad = len(scores["exp_min"][0]), len(scores["exp_min"][1])
    return {"lines": good + bad, "good": good, "bad": bad, **result,
            "best_other_margin": margin, "stop": margin is not None and margin > STOP_MARGIN}


def pins(records: Sequence[dict], taus: Mapping[str, float | None], tolerance: float) -> dict:
    """Per cell, the model lines whose confidence lies further than `tolerance` from tau."""
    out = {}
    for cell, tau in taus.items():
        rows = [r for r in records
                if r["cell"] == cell and r["reader"] == "model" and r["confidence"] is not None]
        off = [] if tau is None else [r for r in rows if abs(r["confidence"] - tau) > tolerance]
        out[cell] = {
            "tau": tau,
            "eligible": len(rows),
            "pinnable": len(off),
            "examples": [
                {k: r[k] for k in ("document", "gt", "text", "confidence")}
                for r in off[:MAX_EXAMPLES]
            ],
        }
    return out
