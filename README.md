# bohnanza-agents

## Capture game events

JSON event logging is opt-in. Configure it before game actions; each emitted
event is one JSON object on one line (JSONL):

```python
from bohnanza import configure_json_logging, disable_json_logging, play_game

configure_json_logging("game.jsonl")
result = play_game(["Agent1", "Agent2"], seed=42)
disable_json_logging()  # optional when the process continues running
```

Call `configure_json_logging()` with no path to write JSON lines to stderr.
Events include game start and score, player hand/field/coin snapshots after
actions, trade offers and responses, and cards transferred. Reconfiguring
logging replaces and closes the previous event handler.

## Run experiments

The built-in batch runner plays reproducible matchups using consecutive seeds:

```python
from bohnanza import run_batch

summary = run_batch(["Agent1", "Agent2"], games=100, seed=1)
print(summary.wins, summary.average_scores, summary.average_turns)
```

Or use the command line summary script. It prints wins, win rates, average
scores, and average turns; `--output` also saves every per-game result and seed
as JSON:

```sh
python scripts/summarize_batch.py --agents Agent1 Agent2 --games 100 --seed 1 \
  --output results/agent1-vs-agent2.json
```

Built-in policy names are `Agent1` and `Agent2`. For example, capture a single
game with `configure_json_logging("game.jsonl")` and call `play_game` as shown
above. The CLI reports aggregate metrics and does not enable event logging.
