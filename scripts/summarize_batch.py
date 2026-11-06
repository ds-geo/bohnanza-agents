#!/usr/bin/env python3
"""Run a seeded agent matchup and print win-rate and score summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bohnanza.sim import run_batch, summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", default=["Agent1", "Agent2"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional JSON results file")
    args = parser.parse_args()
    summary = run_batch(args.agents, games=args.games, seed=args.seed)
    serialized = summary_json(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    concise = {
        "matchup": summary.matchup,
        "games": summary.games,
        "wins": summary.wins,
        "win_rates": {name: count / summary.games for name, count in summary.wins.items()},
        "average_scores": summary.average_scores,
        "average_turns": summary.average_turns,
        "results_file": str(args.output) if args.output else None,
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
