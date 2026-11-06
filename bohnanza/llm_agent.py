"""Prompt templates and state serialization for LLM-backed agents.

Each helper renders the part of ``Player`` or ``Turn`` state relevant to one
decision — ordered hand, bean fields, and the legal actions for that move —
into a compact prompt for an ``LLMClient``. Hands are public information in
Bohnanza, so trade prompts include the other player's hand too.
"""

from __future__ import annotations

from typing import Iterable

from .cards import coins_for
from .player import Player
from .trade import TradeOffer
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
