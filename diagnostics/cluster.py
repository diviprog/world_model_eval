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
    """Rank joint (conditions x failure_phase) cells by failure volume.

    Enumerates every combination of ``n_axes`` condition columns, groups by
    those columns plus ``failure_phase``, and ranks the resulting cells by the
    number of failures they contain (volume), with failure rate as a tiebreak.
    Cells below ``min_support`` episodes or ``min_fail`` failures are dropped so
    we don't report noise.

    Returns one row per surfaced cluster with: the axis columns, their values,
    ``failure_phase``, ``n_total`` (episodes in the cell), ``n_fail``, and
    ``fail_rate``.
    """
    if "failure_phase" not in df.columns:
        raise KeyError(
            "df has no 'failure_phase' column; run features.add_failure_phase first"
        )

    df = df.assign(failure=~df["success"])
    rows: list[dict] = []

    for cols in combinations(condition_cols, n_axes):
        keys = list(cols) + ["failure_phase"]
        # Group every episode by (conditions + the phase it broke in). For
        # successes failure_phase is "none", which we exclude below.
        grp = df.groupby(keys, dropna=False).agg(
            n_fail=("failure", "sum"), n_phase=("failure", "size")
        )
        # Episodes in the cell ignoring phase = support for those conditions.
        support = df.groupby(list(cols), dropna=False).size()

        for idx, r in grp.iterrows():
            idx_t = idx if isinstance(idx, tuple) else (idx,)
            *cond_vals, phase = idx_t
            if phase == "none":
                continue
            n_fail = int(r["n_fail"])
            n_total = int(support.loc[tuple(cond_vals) if len(cond_vals) > 1 else cond_vals[0]])
            if n_total < min_support or n_fail < min_fail:
                continue
            row = {col: val for col, val in zip(cols, cond_vals)}
            row["failure_phase"] = phase
            row["n_total"] = n_total
            row["n_fail"] = n_fail
            row["fail_rate"] = n_fail / n_total
            row["_axes"] = cols
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["failure_phase", "n_total", "n_fail", "fail_rate"]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["n_fail", "fail_rate"], ascending=False
    ).reset_index(drop=True)
    return out.head(k)
