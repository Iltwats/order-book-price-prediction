"""Cached-score arithmetic and dev-only coverage guards; no model/data access."""
import json

import numpy as np
import pytest

from scripts.analyze_hidden_gap import analyze


@pytest.fixture
def cached(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"dev": list(range(9)), "confirmation": [99]}}))
    baseline = np.array([-.4, -.2, 0, .2, .4, .6, .8, .9])
    paths = {}
    for name in ("incumbent", "current"):
        t0 = baseline-.01 + (.01 if name == "current" else 0)
        t1 = baseline+.01 + (.03 if name == "current" else 0)
        wp = (t0+t1)/2
        rows = [{"row_group": i, "seq_ix": 100+i, "eligible": True, "t0": float(t0[i]),
                 "t1": float(t1[i]), "weighted_pearson": float(wp[i])} for i in range(8)]
        rows.append({"row_group": 8, "seq_ix": 108, "eligible": False, "t0": None, "t1": None, "weighted_pearson": None})
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"per_sequence": rows, "t0": float(t0.mean()), "t1": float(t1.mean()),
                                    "weighted_pearson": float(wp.mean())}))
        paths[name] = path
    public = {"incumbent": {"score": .3}, "current": {"score": .32}}
    return paths, public, manifest


def test_exact_cached_statistics_and_baseline_quartile_contributions(cached):
    paths, public, manifest = cached
    result = analyze(paths, public, manifest)
    assert result["coverage"] == {"full_dev_sequences": 9, "eligible_sequences": 8, "excluded_sequences": 1}
    base = result["distributions"]["incumbent"]["weighted_pearson"]
    assert base["negative_count"] == 2 and base["zero_count"] == 1
    assert base["bottom_quartile_count"] == base["top_quartile_count"] == 2
    assert base["bottom_quartile_mean"] == pytest.approx(-.3)
    assert base["top_quartile_mean"] == pytest.approx(.85)
    changes = result["paired_vs_incumbent"]["current"]
    assert changes["changes"]["weighted_pearson"]["improved_count"] == 8
    assert changes["changes"]["t0"]["mean"] == pytest.approx(.01)
    assert changes["changes"]["t1"]["mean"] == pytest.approx(.03)
    quarters = changes["by_baseline_wp_quartile"]
    assert [q["count"] for q in quarters] == [2, 2, 2, 2]
    assert sum(q["contribution_to_overall_mean_delta"] for q in quarters) == pytest.approx(.02)
    assert result["offset_summary"]["range"] == pytest.approx(0)
    assert result["new_dataset_rows_read"] == 0 and result["confirmation_used"] is False


@pytest.mark.parametrize("fault", ["confirmation", "identity", "eligibility", "mean"])
def test_rejects_coverage_identity_eligibility_and_aggregate_mismatch(cached, fault):
    paths, public, manifest = cached
    value = json.loads(paths["current"].read_text())
    if fault == "confirmation":
        value["per_sequence"][-1]["row_group"] = 99
    elif fault == "identity":
        value["per_sequence"][0]["seq_ix"] = 999
    elif fault == "eligibility":
        value["per_sequence"][0]["eligible"] = False
    else:
        value["weighted_pearson"] += .01
    paths["current"].write_text(json.dumps(value))
    with pytest.raises(ValueError):
        analyze(paths, public, manifest)
