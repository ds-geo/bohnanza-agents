"""Provider-neutral chat-completions client with retries and usage accounting.

The transport is injectable so agent logic can be exercised without network
calls. API keys are read from the configured environment variable and never
included in logs.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import json
import socket
import time
from pathlib import Path
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .events import log_event


def _dotenv_value(name: str) -> str | None:
    """Read one simple KEY=value entry from a project or working-dir .env."""
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    visited: set[Path] = set()
    for path in candidates:
        if path in visited:
            continue
        visited.add(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            if entry.startswith("export "):
                entry = entry[len("export "):].lstrip()
            key, separator, value = entry.partition("=")
            if not separator or key.strip() != name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    return None


@dataclass(frozen=True)
class LLMConfig:
    """Per-model request and accounting configuration.

    Cost rates are USD per one million tokens. Leave them unset unless the
    current rate for the selected model is known; estimated cost will then be
    ``None`` while token usage is still tracked.
    """

    model: str
    base_url: str
    api_key_env: str = "LLM_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    max_output_tokens: int | None = None
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if not self.api_key_env.strip():
            raise ValueError("api_key_env cannot be empty")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an HTTP or HTTPS URL")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        for rate in (
            self.input_cost_per_million_tokens,
            self.output_cost_per_million_tokens,
        ):
            if rate is not None and rate < 0:
                raise ValueError("token cost rates cannot be negative")


@dataclass(frozen=True)
class LLMResponse:
    """Text and usage returned by a model transport."""

    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMCallResult:
    """Result of one completed model request."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    attempts: int
    latency_seconds: float


class LLMTransport(Protocol):
    """Provider adapter used by :class:`LLMClient`."""

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int | None,
    ) -> LLMResponse: ...


class RetryableLLMError(RuntimeError):
    """A transient provider error that the wrapper may retry."""


class CompatibleHTTPTransport:
    """HTTP adapter for providers exposing an OpenAI-compatible chat API.

    The configured ``base_url`` should be the API root, such as a host ending
    in ``/v1``. The adapter sends a bearer token in ``Authorization`` and posts
    to ``/chat/completions``. Other provider formats can use a custom transport.
    """

    def __init__(self, *, api_key_env: str = "LLM_API_KEY", base_url: str) -> None:
        api_key = os.environ.get(api_key_env) or _dotenv_value(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set the {api_key_env} environment variable"
            )
        self._api_key = api_key
        self._endpoint = base_url.rstrip("/") + "/chat/completions"

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int | None,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        request = Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {408, 409, 429, 500, 502, 503, 504}:
                raise RetryableLLMError(f"HTTP {exc.code}") from exc
            raise RuntimeError(f"LLM provider returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            raise RetryableLLMError(type(exc).__name__) from exc

        try:
            text = body["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise TypeError("message content was not text")
            usage = body.get("usage") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM provider returned an unexpected response") from exc
        return LLMResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )


class LLMClient:
    """Call a configured model with bounded retries and aggregate usage."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: LLMTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = (
            transport if transport is not None
            else CompatibleHTTPTransport(
                api_key_env=config.api_key_env,
                base_url=config.base_url,
            )
        )
        self._sleep = sleep
        self.calls = 0
        self.failed_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_estimated_cost_usd: float | None = (
            0.0 if config.input_cost_per_million_tokens is not None
            and config.output_cost_per_million_tokens is not None else None
        )

    def complete(self, prompt: str) -> LLMCallResult:
        """Submit one prompt, retry transient failures, and record usage."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        start = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self.transport.complete(
                    prompt,
                    model=self.config.model,
                    timeout_seconds=self.config.timeout_seconds,
                    max_output_tokens=self.config.max_output_tokens,
                )
                break
            except RetryableLLMError as exc:
                if attempts > self.config.max_retries:
                    latency = time.monotonic() - start
                    self.failed_calls += 1
                    log_event("llm.failed", model=self.config.model,
                              attempts=attempts, latency_seconds=latency,
                              error=type(exc).__name__)
                    raise
                delay = self.config.retry_backoff_seconds * (2 ** (attempts - 1))
                self._sleep(delay)
            except Exception as exc:
                latency = time.monotonic() - start
                self.failed_calls += 1
                log_event("llm.failed", model=self.config.model,
                          attempts=attempts, latency_seconds=latency,
                          error=type(exc).__name__)
                raise

        latency = time.monotonic() - start
        cost = self._estimate_cost(response.input_tokens, response.output_tokens)
        self.calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        if cost is not None:
            self.total_estimated_cost_usd = (self.total_estimated_cost_usd or 0.0) + cost
        log_event("llm.completed", model=self.config.model,
                  input_tokens=response.input_tokens,
                  output_tokens=response.output_tokens,
                  estimated_cost_usd=cost, attempts=attempts,
                  latency_seconds=latency)
        return LLMCallResult(
            text=response.text,
            model=self.config.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost_usd=cost,
            attempts=attempts,
            latency_seconds=latency,
        )

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        input_rate = self.config.input_cost_per_million_tokens
        output_rate = self.config.output_cost_per_million_tokens
        if input_rate is None or output_rate is None:
            return None
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
