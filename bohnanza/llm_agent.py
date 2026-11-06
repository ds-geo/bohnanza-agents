"""Prompt templates and state serialization for LLM-backed agents.

Each helper renders the part of ``Player`` or ``Turn`` state relevant to one
decision — ordered hand, bean fields, and the legal actions for that move —
into a compact prompt for an ``LLMClient``. Hands are public information in
Bohnanza, so trade prompts include the other player's hand too.

``LLMAgent`` sends those prompts, parses the JSON reply, and validates the
chosen move against the game state. Unparseable or illegal answers fall back
to a rule-based policy so a bad reply can never break the engine.

``PolicyNegotiationAgent`` (Agent 3) narrows that to negotiation only: the
hand-crafted Agent 2 policy plants and harvests, and the model is asked solely
to propose and answer trades.
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, TypeVar

from .agents import Agent, RuleBasedAgent, StrategicPolicyAgent
from .cards import coins_for
from .events import log_event
from .llm import LLMClient, LLMConfig
from .player import Player
from .trade import TradeOffer, TradeValidator
from .turn import Turn

RULES_BRIEF = (
    "You are playing Bohnanza, a bean-trading card game. Your hand is ordered "
    "and only the front card may be planted; a bean can only be planted in a "
    "field that is empty or already grows that bean, otherwise a field must "
    "be harvested first. Harvesting a field converts its cards to coins, and "
    "the player with the most coins wins."
)

OUTPUT_INSTRUCTION = "Reply with only a JSON object: no prose, no code fences."


def serialize_player(player: Player, pending: Iterable[str] = ()) -> dict:
    """A JSON-friendly snapshot of one player's public state."""

    return {
        "name": player.name,
        "coins": player.coins,
        "hand": list(player.hand),
        "fields": [
            {
                "bean_type": bean_field.bean_type,
                "count": len(bean_field.beans),
                "coins_now": (
                    coins_for(bean_field.bean_type, len(bean_field.beans))
                    if bean_field.bean_type
                    else 0
                ),
            }
            for bean_field in player.fields
        ],
        "pending": list(pending),
    }


def serialize_turn(turn: Turn) -> dict:
    """A JSON-friendly snapshot of the current turn, all players included."""

    return {
        "phase": turn.phase.value,
        "active_index": turn.active_index,
        "face_up": list(turn.face_up),
        "players": [
            serialize_player(player, turn.pending.get(index, []))
            for index, player in enumerate(turn.players)
        ],
    }


def plant_prompt(player: Player, bean: str) -> str:
    """Prompt asking which field index should receive the front hand card."""

    legal = player.legal_fields(bean)
    legal_text = (
        f"Legal field indices for {bean}: {legal}"
        if legal
        else f"No field accepts {bean}; a field must be harvested first."
    )
    return "\n".join(
        [
            RULES_BRIEF,
            "",
            f'You are "{player.name}".',
            *_state_lines(player),
            "",
            f"Plant the front card of your hand: {bean}.",
            legal_text,
            '{"field": <field index>}',
            OUTPUT_INSTRUCTION,
        ]
    )


def harvest_prompt(player: Player) -> str:
    """Prompt asking which occupied field to harvest for coins."""

    occupied = [
        index for index, bean_field in enumerate(player.fields) if bean_field.beans
    ]
    return "\n".join(
        [
            RULES_BRIEF,
            "",
            f'You are "{player.name}".',
            *_state_lines(player),
            "",
            "Choose a field to harvest into coins.",
            f"Occupied field indices: {occupied}",
            '{"field": <field index>}',
            OUTPUT_INSTRUCTION,
        ]
    )


def propose_prompt(turn: Turn, responder_index: int) -> str:
    """Prompt asking the active player for a trade offer to the responder."""

    proposer = turn.active_player
    responder = turn.players[responder_index]
    return "\n".join(
        [
            RULES_BRIEF,
            "",
            f'You are "{proposer.name}" (player {turn.active_index}), '
            f'offering a trade to "{responder.name}" (player {responder_index}).',
            *_state_lines(proposer),
            "",
            f"Face-up cards to trade away: {_indexed(turn.face_up)}",
            f'"{responder.name}" fields:',
            *_field_lines(responder),
            f'"{responder.name}" hand: {_indexed(responder.hand)}',
            "",
            "Name the card indices you give (from your hand or the face-up "
            "cards) and the indices of responder hand cards you want in return.",
            '{"pass": true} or {"give": [<indices>], '
            '"give_source": "hand"|"face_up", "take": [<indices>]}',
            OUTPUT_INSTRUCTION,
        ]
    )


def respond_prompt(turn: Turn, offer: TradeOffer) -> str:
    """Prompt asking the responder to accept or reject a pending offer."""

    responder = turn.players[offer.responder]
    source = (
        turn.face_up if offer.proposer_face_up else turn.players[offer.proposer].hand
    )
    gives = [source[i] for i in offer.proposer_gives if i < len(source)]
    takes = [
        responder.hand[i] for i in offer.responder_gives if i < len(responder.hand)
    ]
    return "\n".join(
        [
            RULES_BRIEF,
            "",
            f'You are "{responder.name}" (player {offer.responder}).',
            *_state_lines(responder),
            "",
            f'Player {offer.proposer} offers you {gives or "nothing"} '
            f"in exchange for {takes or 'nothing'} from your hand.",
            "Accept or reject the offer.",
            '{"accept": true} or {"accept": false}',
            OUTPUT_INSTRUCTION,
        ]
    )


def _state_lines(player: Player) -> list[str]:
    return [
        f"Hand (front first): {_indexed(player.hand)}",
        "Fields:",
        *_field_lines(player),
        f"Coins: {player.coins}",
    ]


def _field_lines(player: Player) -> list[str]:
    lines = []
    for index, bean_field in enumerate(player.fields):
        if bean_field.bean_type is None:
            lines.append(f"  {index}: empty")
        else:
            coins = coins_for(bean_field.bean_type, len(bean_field.beans))
            lines.append(
                f"  {index}: {bean_field.bean_type} x{len(bean_field.beans)} "
                f"(worth {coins} coins now)"
            )
    return lines


def _indexed(cards: list[str]) -> str:
    return ", ".join(f"{index}={card}" for index, card in enumerate(cards)) or "none"


class LLMParseError(ValueError):
    """The model's reply contained no usable JSON object."""


def parse_action(text: str) -> dict:
    """Extract the decision object from a model reply.

    Tolerates code fences and surrounding prose by scanning for the first
    balanced ``{...}`` span that parses as JSON. Raises ``LLMParseError``
    when nothing parses to an object.
    """

    cleaned = _unfence(text).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = _extract_object(cleaned)
    if not isinstance(value, dict):
        raise LLMParseError(f"Expected a JSON object, got {value!r}")
    return value


def _unfence(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("```")
    )


def _extract_object(text: str) -> object:
    """Return the first balanced ``{...}`` span that parses as JSON."""

    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for position, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = position
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : position + 1])
                except json.JSONDecodeError:
                    start = None  # keep scanning for the next candidate
    raise LLMParseError("No JSON object found in the response")


_T = TypeVar("_T")


def default_client() -> LLMClient:
    """A client for the model named by the LLM_MODEL/LLM_BASE_URL variables."""

    return LLMClient(LLMConfig.from_env())


class LLMAgent(Agent):
    """An agent that asks an ``LLMClient`` for each decision.

    Every reply is parsed and checked against the legal moves in the current
    state; a transport failure, an unparseable reply, or an illegal move is
    delegated to ``fallback`` (a ``RuleBasedAgent`` by default) and logged as
    an ``llm.fallback`` event.
    """

    def __init__(self, client: LLMClient, fallback: Agent | None = None) -> None:
        self.client = client
        self.fallback = fallback if fallback is not None else RuleBasedAgent()

    def plant(self, player: Player, bean: str) -> int:
        try:
            action = self._ask("plant", plant_prompt(player, bean))
            index = _int_value(action, "field")
            if index not in player.legal_fields(bean):
                raise ValueError(f"Field {index} does not accept {bean}")
            return index
        except Exception as exc:
            return self._recover("plant", exc, self.fallback.plant, player, bean)

    def propose(self, turn: Turn, responder_index: int) -> TradeOffer | None:
        try:
            action = self._ask(
                "propose", propose_prompt(turn, responder_index)
            )
            return self._offer_from_action(turn, responder_index, action)
        except Exception as exc:
            return self._recover(
                "propose", exc, self.fallback.propose, turn, responder_index
            )

    def respond(self, turn: Turn, offer: TradeOffer) -> bool:
        try:
            action = self._ask("respond", respond_prompt(turn, offer))
            accept = action.get("accept")
            if not isinstance(accept, bool):
                raise ValueError(f"Expected a boolean 'accept', got {accept!r}")
            return accept
        except Exception as exc:
            return self._recover("respond", exc, self.fallback.respond, turn, offer)

    def harvest(self, player: Player) -> int | None:
        try:
            action = self._ask("harvest", harvest_prompt(player))
            index = _int_value(action, "field")
            if index >= len(player.fields) or not player.fields[index].beans:
                raise ValueError(f"Field {index} is not an occupied field")
            return index
        except Exception as exc:
            return self._recover("harvest", exc, self.fallback.harvest, player)

    def _ask(self, decision: str, prompt: str) -> dict:
        action = parse_action(self.client.complete(prompt).text)
        log_event("llm.decision", decision=decision, action=action)
        return action

    def _recover(
        self,
        decision: str,
        error: Exception,
        fallback: Callable[..., _T],
        *args: object,
    ) -> _T:
        log_event(
            "llm.fallback",
            decision=decision,
            error=type(error).__name__,
            detail=str(error),
        )
        return fallback(*args)

    @staticmethod
    def _offer_from_action(
        turn: Turn, responder_index: int, action: dict
    ) -> TradeOffer | None:
        """Validate an offer payload before building it, so rejected shapes
        never reach the ``trade.offered`` event stream."""

        if action.get("pass"):
            return None
        give = _int_list(action, "give")
        take = _int_list(action, "take")
        if not give and not take:
            raise ValueError("Offer must give or take at least one card")
        source = action.get("give_source", "hand")
        if source not in ("hand", "face_up"):
            raise ValueError(f"Unknown give_source: {source!r}")
        if not give:
            source = "hand"  # giving nothing makes the source moot

        proposer = turn.active_index
        face_up = source == "face_up"
        give_bound = len(turn.face_up) if face_up else len(turn.players[proposer].hand)
        _check_indices(give, give_bound, "give")
        _check_indices(take, len(turn.players[responder_index].hand), "take")

        offer = TradeOffer(
            proposer=proposer,
            responder=responder_index,
            proposer_gives=give,
            responder_gives=take,
            proposer_face_up=face_up,
        )
        valid, reason = TradeValidator(
            turn.players, turn.active_index, len(turn.face_up)
        ).validate(offer)
        if not valid:
            raise ValueError(reason)
        return offer


def _int_value(action: dict, key: str) -> int:
    raw = action.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"Expected an integer for {key!r}, got {raw!r}")
    if raw < 0:
        raise ValueError(f"Expected a non-negative index for {key!r}")
    return raw


def _int_list(action: dict, key: str) -> list[int]:
    raw = action.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list for {key!r}, got {raw!r}")
    return [
        _int_value({key: index}, key) for index in raw
    ]


def _check_indices(indices: list[int], bound: int, side: str) -> None:
    for index in indices:
        if index >= bound:
            raise ValueError(f"{side} index {index} is out of range")


class PolicyNegotiationAgent(LLMAgent):
    """A hand-crafted policy that negotiates through a model (Agent 3).

    Planting and harvesting stay with ``StrategicPolicyAgent``, which already
    plays them well and costs nothing; only the open-ended half of the game,
    proposing and answering trades, is asked of the model. The same policy is
    the fallback, so a failed or illegal reply plays as Agent 2 would.
    """

    def __init__(
        self, client: LLMClient | None = None, policy: Agent | None = None
    ) -> None:
        super().__init__(
            client if client is not None else default_client(),
            fallback=policy if policy is not None else StrategicPolicyAgent(),
        )

    def plant(self, player: Player, bean: str) -> int:
        return self.fallback.plant(player, bean)

    def harvest(self, player: Player) -> int | None:
        return self.fallback.harvest(player)


# Short alias matching the Agent1/Agent2 policy names.
Agent3 = PolicyNegotiationAgent
