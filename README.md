# bohnanza-agents

## View JSON game logs

Enable structured event output before running your game:

```python
from bohnanza import configure_json_logging

configure_json_logging()  # JSON lines appear in the terminal (stderr)
```

To save events to a file instead:

```python
configure_json_logging("game.jsonl")
```

Each line in `game.jsonl` is a separate JSON event. Call the setup once near
the start of your game script, before taking game actions.
