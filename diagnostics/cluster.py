"""Pattern surfacing: find the dominant failure clusters via groupby.

Nothing fancier than a groupby. A table that says "85% failure on
reflective-texture scenes with a distractor, almost all during grasp" is more
legible to a founder than a t-SNE plot, and you can debug it. Embedding-based
clustering is deliberately out of scope on the first pass (SCOPE.md): only reach
for it if the structured cut leaves a meaningful chunk of failures unexplained.

Everything here returns small ranked tables, not models.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from diagnostics.schema import CONDITION_AXES


def overall_success_rate(df: pd.DataFrame) -> float:
    """The headline number everyone already reports."""
    return float(df["success"].mean())


def marginal_failure_rates(
    df: pd.DataFrame, condition_cols: tuple[str, ...] = CONDITION_AXES
) -> dict[str, pd.DataFrame]:
    """Per-axis failure rate: for each condition column, the failure rate at
    each of its values. Shows which single axis moves the needle."""
    out: dict[str, pd.DataFrame] = {}
    for col in condition_cols:
        g = (
            df.assign(failure=~df["success"])
            .groupby(col, dropna=False)["failure"]
            .agg(n="count", n_fail="sum")
            .reset_index()
        )
        g["fail_rate"] = g["n_fail"] / g["n"]
        out[col] = g.sort_values("fail_rate", ascending=False).reset_index(drop=True)
    return out


def phase_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Failure counts by failure phase (failed episodes only). Requires the
    ``failure_phase`` column from ``features.add_failure_phase``."""
    fails = df[~df["success"]]
    g = (
        fails.groupby("failure_phase")
        .size()
        .reset_index(name="n_fail")
        .sort_values("n_fail", ascending=False)
        .reset_index(drop=True)
    )
    return g


def top_failure_clusters(
    df: pd.DataFrame,
    condition_cols: tuple[str, ...] = CONDITION_AXES,
    n_axes: int = 2,
    min_support: int = 15,
    min_fail: int = 8,
    k: int = 5,
) -> pd.DataFrame:
    """Surface the condition clusters that fail the most *above baseline*.

    Enumerates every combination of ``n_axes`` condition columns and groups by
    those columns. Each cell is scored by **failure excess** — how many more
    episodes failed than the overall failure rate would predict
    (``n_fail - n_total * overall_failure_rate``). This balances volume against
    severity: a small cell that fails 93% of the time and a large cell that fails
    well above average both rank highly, while a large cell that fails *below*
    average (e.g. the default setup) is correctly pushed down.

    Phase is not part of the grouping key — instead each surfaced cluster is
    annotated with the **dominant failure phase among its failures**, which reads
    as "these conditions fail X% of the time, mostly at the <phase> phase."

    Cells below ``min_support`` episodes or ``min_fail`` failures are dropped.
    Returns one row per cluster with the axis columns + values, ``failure_phase``
    (dominant), ``n_total``, ``n_fail``, ``fail_rate``, and ``excess``.
    """
    if "failure_phase" not in df.columns:
        raise KeyError(
            "df has no 'failure_phase' column; run features.add_failure_phase first"
        )

    df = df.assign(failure=~df["success"])
    overall_fail = float(df["failure"].mean())
    rows: list[dict] = []

    for cols in combinations(condition_cols, n_axes):
        grp = df.groupby(list(cols), dropna=False).agg(
            n_total=("failure", "size"), n_fail=("failure", "sum")
        )
        for idx, r in grp.iterrows():
            idx_t = idx if isinstance(idx, tuple) else (idx,)
            n_total, n_fail = int(r["n_total"]), int(r["n_fail"])
            if n_total < min_support or n_fail < min_fail:
                continue
            # dominant failure phase among this cell's failures
            mask = pd.Series(True, index=df.index)
            for c, v in zip(cols, idx_t):
                mask &= df[c] == v
            phases = df.loc[mask & df["failure"], "failure_phase"]
            dom_phase = phases.mode().iat[0] if len(phases) else "n/a"
            row = {col: val for col, val in zip(cols, idx_t)}
            row["failure_phase"] = dom_phase
            row["n_total"] = n_total
            row["n_fail"] = n_fail
            row["fail_rate"] = n_fail / n_total
            row["excess"] = n_fail - n_total * overall_fail
            row["_axes"] = cols
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["failure_phase", "n_total", "n_fail", "fail_rate", "excess"]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["excess", "fail_rate"], ascending=False).reset_index(drop=True)
    return out.head(k)
