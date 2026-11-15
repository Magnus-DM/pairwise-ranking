import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker import Dataset, Ranker  # noqa: E402


def make_ranker(tmp_path, n=8):
    data = tmp_path / "items.csv"
    data.write_text("id,name\n" + "".join(f"i{k},Item {k}\n" for k in range(n)))
    return Ranker(Dataset(data), str(tmp_path / "prefs.sqlite"))


def test_mm_fit_recovers_known_strengths(tmp_path):
    """With many simulated choices, fitted log-strengths track the true ones."""
    rng = random.Random(0)
    n = 8
    true = [0.6 * k for k in range(n)]  # i7 strongest, i0 weakest
    ranker = make_ranker(tmp_path, n)
    for _ in range(3000):
        i, j = rng.sample(range(n), 2)
        p_i = 1 / (1 + math.exp(true[j] - true[i]))
        ranker.vote(f"i{i}", f"i{j}", 1 if rng.random() < p_i else 0)

    ranking = ranker.ranking()
    assert [row["id"] for row in ranking] == [f"i{k}" for k in reversed(range(n))]

    fitted = ranker.logp
    gaps_true = [true[k + 1] - true[k] for k in range(n - 1)]
    gaps_fit = [fitted[k + 1] - fitted[k] for k in range(n - 1)]
    # The prior (one virtual win and loss vs. an average opponent) shrinks the
    # scale a little, so compare the overall spread with a loose tolerance.
    assert sum(gaps_fit) == pytest.approx(sum(gaps_true), rel=0.15)


def test_unbeaten_entry_stays_finite(tmp_path):
    ranker = make_ranker(tmp_path, 3)
    for _ in range(20):
        ranker.vote("i0", "i1", 1)
        ranker.vote("i0", "i2", 1)
    top = ranker.ranking()[0]
    assert top["id"] == "i0"
    assert math.isfinite(ranker.logp[0])


def test_choices_persist_and_undo(tmp_path):
    ranker = make_ranker(tmp_path, 4)
    ranker.vote("i0", "i1", 1)
    ranker.vote("i2", "i3", 0.5)
    assert ranker.stats()["comparisons"] == 2

    undone, pair = ranker.undo()
    assert undone and {c["id"] for c in pair} == {"i2", "i3"}

    reopened = Ranker(ranker.ds, str(tmp_path / "prefs.sqlite"))
    assert reopened.stats()["comparisons"] == 1


def test_invalid_votes_are_rejected(tmp_path):
    ranker = make_ranker(tmp_path, 3)
    with pytest.raises(ValueError):
        ranker.vote("i0", "i0", 1)
    with pytest.raises(ValueError):
        ranker.vote("i0", "missing", 1)
    with pytest.raises(ValueError):
        ranker.vote("i0", "i1", 0.3)


def test_next_pair_returns_two_distinct_entries(tmp_path):
    random.seed(1)
    ranker = make_ranker(tmp_path, 10)
    for _ in range(50):
        a, b = ranker.next_pair()
        assert a["id"] != b["id"]
        ranker.vote(a["id"], b["id"], 1)


def test_json_dataset_with_id_mapping(tmp_path):
    data = tmp_path / "items.json"
    data.write_text(json.dumps({"x": {"name": "X"}, "y": {"name": "Y"}}))
    ds = Dataset(data)
    assert ds.order == ["x", "y"]
    assert ds.items["x"]["title"] == "X"
