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

Built-in policy names are `Agent1`, `Agent2`, and `Agent3`. `Agent3` plants and
harvests with the `Agent2` policy but asks a model to propose and answer trades,
so running it needs the LLM setup below and makes a paid API call per trade
decision. For example, capture a single
game with `configure_json_logging("game.jsonl")` and call `play_game` as shown
above. The CLI reports aggregate metrics and does not enable event logging.

## LLM client setup

The built-in transport uses Python's standard library, so there is no provider
SDK to install. Create an API key with a provider that supports the
OpenAI-compatible Chat Completions format, then put the key in `.env` or set it
in your shell. The client checks the process environment first, then looks for
`.env` in the working directory or repository root. Keep keys out of source
files and logs:

```sh
export LLM_API_KEY="your-provider-key"
```

You can instead store it in `.env` as `LLM_API_KEY=your-provider-key`; you do
not need to source the file manually.

Agents that call a model, such as `Agent3`, also need a model name and API root.
Both are required and are read from the same places as the key; building the
agent fails with an error naming the missing variable. A Gemini example using
its OpenAI-compatible endpoint:

```sh
LLM_API_KEY=your-gemini-key
LLM_MODEL=gemini-flash-latest
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
```

The client takes an explicit model name, API base URL, and optional
per-million-token rates. The default transport posts to
`{base_url}/chat/completions` with a bearer key. It retries transient
connection, timeout, rate-limit, and server errors. Calls emit token, cost,
attempt, and latency metadata through the opt-in JSON event logger; prompts,
responses, and API keys are not logged.

```python
from bohnanza import LLMClient, LLMConfig, configure_json_logging

configure_json_logging("llm-calls.jsonl")
client = LLMClient(LLMConfig(
    model="your-provider-model-name",
    base_url="https://provider.example/v1",
))
reply = client.complete("Return a short greeting.")
print(reply.text, reply.input_tokens, reply.output_tokens)
```

Cost estimates are available when both token rates are supplied in
`LLMConfig`; check the current model pricing before configuring them. The
transport can be injected into `LLMClient` for offline or mock usage. If a
provider uses a different request or response format, implement the
`LLMTransport` protocol and pass it to `LLMClient`.
