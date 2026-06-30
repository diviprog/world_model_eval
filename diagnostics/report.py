"""The one-page diagnostic — the actual deliverable.

Leads with the headline success rate (the number they already expect), then the
decomposition they don't have: failure breakdown by condition and by phase. For
each dominant cluster it emits a concrete data-collection recommendation. The
output is plain Markdown so it drops straight into an outreach email or a repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from diagnostics import cluster, features, schema
from diagnostics.schema import CONDITION_AXES

# Human-readable descriptions of each condition value, for recommendation text.
_PHASE_GLOSS = {
    "reach": "reaching toward the object",
    "grasp": "closing the gripper on the object",
    "transport": "lifting and carrying the object",
    "place": "releasing the object at the target",
}


def _describe_conditions(row: pd.Series, axes: tuple[str, ...]) -> str:
    parts = []
    for ax in axes:
        if ax in row and pd.notna(row[ax]):
            val = row[ax]
            if ax == "distractor":
                parts.append("with a distractor present" if val else "with no distractor")
            else:
                parts.append(f"{val} {ax.replace('_', ' ')}")
    return ", ".join(parts)


def _recommendation(row: pd.Series, axes: tuple[str, ...]) -> str:
    cond = _describe_conditions(row, axes)
    phase = row["failure_phase"]
    gloss = _PHASE_GLOSS.get(phase, phase)
    return (
        f"Collect demonstrations in **{cond}** scenes, targeting the "
        f"**{phase}** phase ({gloss}). This cluster fails "
        f"{row['fail_rate']:.0%} of the time across {int(row['n_fail'])} "
        f"failed episodes — the single largest recoverable gap."
    )


def generate_report(df: pd.DataFrame) -> str:
    """Build the Markdown diagnostic from a frame of rollout records.

    The frame must already carry a ``failure_phase`` column (run
    ``features.add_failure_phase`` first); ``main`` does this for you.
    """
    policy = df["policy"].iloc[0]
    task = df["task"].iloc[0]
    n = len(df)
    success_rate = cluster.overall_success_rate(df)

    lines: list[str] = []
    lines.append(f"# Failure Diagnostic — `{policy}` on `{task}`")
    lines.append("")
    lines.append(f"**{n} episodes** · variant-aggregation sweep")
    lines.append("")

    # 1. The headline they already expect.
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **Success rate: {success_rate:.1%}**  "
        f"({int(df['success'].sum())}/{n} episodes)"
    )
    lines.append("")

    # 2. The decomposition they don't have — by phase.
    phase_tbl = cluster.phase_distribution(df)
    total_fail = int((~df["success"]).sum())
    lines.append("## Where it breaks (failure phase)")
    lines.append("")
    lines.append("| Phase | Failed episodes | Share of failures |")
    lines.append("|---|---:|---:|")
    for _, r in phase_tbl.iterrows():
        share = r["n_fail"] / total_fail if total_fail else 0.0
        lines.append(f"| {r['failure_phase']} | {int(r['n_fail'])} | {share:.0%} |")
    lines.append("")

    # 3. By condition — which axes move the needle.
    lines.append("## What drives failure (by condition)")
    lines.append("")
    marg = cluster.marginal_failure_rates(df)
    for ax in CONDITION_AXES:
        tbl = marg[ax]
        worst = tbl.iloc[0]
        lines.append(
            f"- **{ax.replace('_', ' ')}:** worst at "
            f"`{worst[ax]}` — {worst['fail_rate']:.0%} failure "
            f"({int(worst['n_fail'])}/{int(worst['n'])})"
        )
    lines.append("")

    # 4. Dominant clusters + the actionable recommendation.
    clusters = cluster.top_failure_clusters(df)
    lines.append("## Dominant failure clusters & what to collect")
    lines.append("")
    if clusters.empty:
        lines.append("_No cluster cleared the support threshold._")
    else:
        for i, (_, row) in enumerate(clusters.iterrows(), 1):
            axes = row["_axes"]
            cond = _describe_conditions(row, axes)
            lines.append(
                f"{i}. **{cond} → {row['failure_phase']}** "
                f"({row['fail_rate']:.0%} failure, {int(row['n_fail'])}/{int(row['n_total'])} episodes)"
            )
        lines.append("")
        lines.append("### Recommendation")
        lines.append("")
        lines.append(_recommendation(clusters.iloc[0], clusters.iloc[0]["_axes"]))
    lines.append("")

    return "\n".join(lines)


def build_frame(path: str | Path | None) -> pd.DataFrame:
    """Load records from ``path`` (JSONL), or fall back to synthetic fixtures."""
    if path is not None and Path(path).exists():
        df = schema.load_jsonl(path)
    else:
        from diagnostics import fixtures

        df = fixtures.generate_frame()
    return features.add_failure_phase(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the one-page diagnostic.")
    parser.add_argument(
        "--data",
        default="data/rollouts.jsonl",
        help="JSONL rollout records (falls back to synthetic fixtures if absent)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the Markdown report here instead of stdout",
    )
    args = parser.parse_args(argv)

    df = build_frame(args.data)
    report = generate_report(df)

    if not Path(args.data).exists():
        report = (
            "> ⚠️ Generated from **synthetic fixtures** — no rollout data found "
            f"at `{args.data}`.\n\n" + report
        )

    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
