"""Build demo/demo.sqlite: simulated choices on the made-up roles in demo/roles.csv.

Each role gets a hidden "true" log-strength drawn from N(0, 1.5^2). The script
lets the app pick pairs exactly as it would for a real user, answers each one
by sampling from the Bradley-Terry model with those hidden strengths, and
stores the choices.
At the end it prints how well the fitted ranking recovers the hidden order.

    python3 demo/make_demo_db.py            # 150 choices, seed 7
    python3 demo/make_demo_db.py --n 400 --seed 1
"""

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ranker import Dataset, Ranker  # noqa: E402


def spearman(x, y):
    """Spearman rank correlation for two equal-length lists without ties."""
    rx = {v: r for r, v in enumerate(sorted(range(len(x)), key=x.__getitem__))}
    ry = {v: r for r, v in enumerate(sorted(range(len(y)), key=y.__getitem__))}
    n = len(x)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "demo" / "roles.csv"))
    ap.add_argument("--db", default=str(ROOT / "demo" / "demo.sqlite"))
    ap.add_argument("--n", type=int, default=150, help="number of simulated choices")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--tie-margin", type=float, default=0.03, help="P(win) within 0.5 +/- this is a tie"
    )
    args = ap.parse_args()

    db = Path(args.db)
    if db.exists():
        db.unlink()

    random.seed(args.seed)  # the app's pair picker uses the global random module
    rng = random.Random(args.seed)
    ds = Dataset(args.data)
    hidden = {iid: rng.gauss(0.0, 1.5) for iid in ds.order}
    ranker = Ranker(ds, str(db))

    for _ in range(args.n):
        a, b = (card["id"] for card in ranker.next_pair())
        p_a = 1.0 / (1.0 + math.exp(hidden[b] - hidden[a]))
        if abs(p_a - 0.5) < args.tie_margin:
            result = 0.5
        else:
            result = 1 if rng.random() < p_a else 0
        ranker.vote(a, b, result)

    ranking = ranker.ranking()
    fitted = {row["id"]: -row["rank"] for row in ranking}
    rho = spearman([hidden[i] for i in ds.order], [fitted[i] for i in ds.order])
    print(f"Wrote {args.n} simulated choices to {db}")
    print(f"Spearman correlation between hidden and fitted order: {rho:.3f}")


if __name__ == "__main__":
    main()
