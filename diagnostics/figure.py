"""Render the diagnostic as a single shareable figure — the outreach artifact.

Two panels: success rate by the condition axes that move the needle (colored
red->green by rate), and the failure-phase decomposition. This is the screenshot
that goes in the email; the numbers come straight from the same diagnostics path
as the Markdown report.

matplotlib is an optional extra (not needed by the core diagnostics package).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from diagnostics import cluster, features, schema
from diagnostics.schema import CONDITION_AXES


def render(df: pd.DataFrame, out_path: str | Path, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import RdYlGn

    policy = df["policy"].iloc[0]
    task = df["task"].iloc[0]
    overall = cluster.overall_success_rate(df)

    # Left panel: success rate per value of each condition axis (sorted worst->best).
    marg = cluster.marginal_failure_rates(df)
    bars = []  # (label, success_rate, n)
    for ax in CONDITION_AXES:
        t = marg[ax]
        for _, r in t.iterrows():
            bars.append((f"{ax.replace('_',' ')}={r[ax]}", 1 - r["fail_rate"], int(r["n"])))
    bars.sort(key=lambda b: b[1])  # worst success first
    labels = [b[0] for b in bars]
    rates = [b[1] for b in bars]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [2.0, 1.0]}
    )
    colors = [RdYlGn(r) for r in rates]
    axL.barh(range(len(labels)), rates, color=colors, edgecolor="#333", linewidth=0.4)
    axL.set_yticks(range(len(labels)))
    axL.set_yticklabels(labels, fontsize=8)
    axL.axvline(overall, color="#222", ls="--", lw=1)
    axL.text(overall, len(labels) - 0.3, f" overall {overall:.0%}", fontsize=8, va="top")
    axL.set_xlim(0, 1)
    axL.set_xlabel("success rate")
    axL.set_title("Success rate by condition", fontsize=11, loc="left")
    for i, (r, n) in enumerate((b[1], b[2]) for b in bars):
        axL.text(min(r + 0.01, 0.95), i, f"{r:.0%} (n={n})", va="center", fontsize=7)

    # Right panel: failure-phase decomposition.
    ph = cluster.phase_distribution(df)
    order = ["reach", "grasp", "transport", "place"]
    ph = ph.set_index("failure_phase").reindex(order).dropna()
    phase_colors = {"reach": "#6baed6", "grasp": "#fb6a4a", "transport": "#fdae6b", "place": "#74c476"}
    axR.bar(ph.index, ph["n_fail"], color=[phase_colors[p] for p in ph.index], edgecolor="#333")
    axR.set_ylabel("failed episodes")
    axR.set_title("Where it breaks (phase)", fontsize=11, loc="left")
    total_fail = int(ph["n_fail"].sum())
    for p, v in ph["n_fail"].items():
        axR.text(p, v, f"{int(v)}\n{v/total_fail:.0%}", ha="center", va="bottom", fontsize=8)

    sup = title or f"Failure diagnostic — {policy} on {task}   |   {len(df)} episodes, {overall:.0%} success"
    fig.suptitle(sup, fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the diagnostic figure.")
    ap.add_argument("--data", default="data/rollouts.jsonl")
    ap.add_argument("--out", default="results/diagnostic.png")
    args = ap.parse_args(argv)
    df = features.add_failure_phase(schema.load_jsonl(args.data))
    render(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
