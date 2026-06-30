"""Gate A: the diagnostics pipeline recovers the planted failure pattern.

Fixtures plant a single dominant failure cluster — grasp-phase failures on
reflective-texture scenes with a distractor present. If features -> cluster
can't surface that, a higher layer is wrong, and we want to know before any real
rollout exists.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from diagnostics import cluster, features, fixtures, schema, report


@pytest.fixture(scope="module")
def tagged() -> pd.DataFrame:
    df = fixtures.generate_frame(n=600, seed=7)
    return features.add_failure_phase(df)


def test_records_validate():
    for rec in fixtures.generate_records(n=50, seed=1):
        schema.validate(rec)  # raises on nonconformance


def test_jsonl_roundtrip(tmp_path):
    recs = fixtures.generate_records(n=30, seed=2)
    path = tmp_path / "rollouts.jsonl"
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    df = schema.load_jsonl(path)
    assert len(df) == 30
    # conditions flattened into columns
    for ax in schema.CONDITION_AXES:
        assert ax in df.columns
    assert not df[list(schema.CONDITION_AXES)].isna().any().any()


def test_validate_rejects_length_mismatch():
    rec = fixtures.generate_records(n=1, seed=3)[0]
    rec["trajectory"]["gripper_state"].append(0)  # break the length contract
    with pytest.raises(ValueError):
        schema.validate(rec)


def test_failure_phase_values(tagged):
    valid = {"none", *features.PHASES}
    assert set(tagged["failure_phase"]).issubset(valid)
    # Successful episodes are tagged "none"; failures never are.
    assert (tagged.loc[tagged["success"], "failure_phase"] == "none").all()
    assert (tagged.loc[~tagged["success"], "failure_phase"] != "none").all()


def test_planted_cluster_is_top(tagged):
    clusters = cluster.top_failure_clusters(tagged)
    assert not clusters.empty
    top = clusters.iloc[0]
    # The planted pattern: reflective + distractor, breaking at grasp.
    assert top["failure_phase"] == "grasp"
    assert top.get("table_texture") == "reflective"
    assert bool(top.get("distractor")) is True
    # And it should be a high-rate cell, not just incidental volume.
    assert top["fail_rate"] >= 0.6


def test_planted_grasp_failures_concentrate(tagged):
    # Grasp failures should be concentrated in the reflective+distractor cell.
    grasp_fail = tagged[(~tagged["success"]) & (tagged["failure_phase"] == "grasp")]
    in_cluster = grasp_fail[
        (grasp_fail["table_texture"] == "reflective") & (grasp_fail["distractor"])
    ]
    assert len(in_cluster) / len(grasp_fail) >= 0.5


def test_report_answers_the_four_questions(tagged):
    md = report.generate_report(tagged)
    # 1. headline rate, 2. by-phase, 3. by-condition, 4. what to collect
    assert "Success rate:" in md
    assert "failure phase" in md.lower()
    assert "by condition" in md.lower()
    assert "Recommendation" in md
    assert "grasp" in md
