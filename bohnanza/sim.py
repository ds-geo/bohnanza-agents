"""Seeded simulation helpers for running agent matchups."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Sequence

from .agents import Agent, Agent1, Agent2
from .events import log_event
from .game import Game, GameConfig, GameResult
from .llm_agent import Agent3, Agent4
from .trade import TradeValidator


AGENT_TYPES: dict[str, type[Agent]] = {
    "Agent1": Agent1,
    "Agent2": Agent2,
    "Agent3": Agent3,
    "Agent4": Agent4,
}


@dataclass
class BatchSummary:
    matchup: list[str]
    games: int
    wins: dict[str, int]
    average_scores: dict[str, float]
    average_turns: float
    results: list[dict]


def play_game(agent_names: Sequence[str], *, seed: int | None = None) -> GameResult:
    """Play one complete game using named built-in policies."""
    if len(agent_names) < 2:
        raise ValueError("A matchup needs at least two agents")
    try:
        agents = [AGENT_TYPES[name]() for name in agent_names]
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {exc.args[0]}") from exc

    player_names = [f"{name}#{index + 1}" for index, name in enumerate(agent_names)]
    game = Game(GameConfig(player_names=player_names, seed=seed))
    log_event("game.started", players=player_names, policies=list(agent_names), seed=seed)
    while not game.is_over:
        turn = game.start_turn()
        player = turn.active_player
        agent = agents[turn.active_index]

        planted = 0
        while player.hand and planted < 2:
            bean = player.hand[0]
            if player.needs_harvest_to_plant(bean):
                field_index = agent.harvest(player)
                if field_index is None:
                    raise RuntimeError("Agent could not free a field for forced planting")
                turn.harvest(turn.active_index, field_index)
            field_index = agent.plant(player, bean)
            turn.plant_from_hand(field_index)
            planted += 1
        turn.end_plant_phase()

        # Each responder may receive at most one offer in this phase. Current
        # built-in policies make direct offers; validation guards future ones.
        for responder_index in range(len(game.players)):
            if responder_index == turn.active_index:
                continue
            offer = agent.propose(turn, responder_index)
            if offer is None:
                continue
            validator = TradeValidator(turn.players, turn.active_index, len(turn.face_up))
            valid, reason = validator.validate(offer)
            responder = agents[responder_index]
            accepted = valid and responder.respond(turn, offer)
            proposer_source = turn.face_up if offer.proposer_face_up else player.hand
            offered_beans = [proposer_source[i] for i in offer.proposer_gives if i < len(proposer_source)]
            requested_beans = [turn.players[responder_index].hand[i]
                               for i in offer.responder_gives
                               if i < len(turn.players[responder_index].hand)]
            log_event(
                "trade.responded",
                proposer=player.name,
                responder=turn.players[responder_index].name,
                proposer_gives=offer.proposer_gives,
                responder_gives=offer.responder_gives,
                offered_beans=offered_beans,
                requested_beans=requested_beans,
                valid=valid,
                validation_error=reason,
                accepted=bool(accepted),
                players=[{
                    "name": p.name, "hand": list(p.hand), "coins": p.coins,
                    "fields": [{"bean_type": f.bean_type, "beans": list(f.beans)} for f in p.fields],
                } for p in turn.players],
            )
            if not valid:
                offer.reject()
            elif accepted:
                offer.accept()
                if offer.proposer_gives:
                    turn.transfer(
                        offer.proposer, offer.responder,
                        face_up_indices=offer.proposer_gives if offer.proposer_face_up else (),
                        hand_indices=() if offer.proposer_face_up else offer.proposer_gives,
                    )
                if offer.responder_gives:
                    turn.transfer(offer.responder, offer.proposer, hand_indices=offer.responder_gives)
            else:
                offer.reject()

        turn.end_trade_phase()
        for index, pending in turn.pending.items():
            while pending:
                bean = pending[0]
                target = agents[index]
                if game.players[index].needs_harvest_to_plant(bean):
                    field_index = target.harvest(game.players[index])
                    if field_index is None:
                        raise RuntimeError("Agent could not free a field for pending planting")
                    turn.harvest(index, field_index)
                field_index = target.plant(game.players[index], bean)
                turn.plant_pending(index, 0, field_index)
        turn.end_plant_traded_phase()
        turn.draw_three()
        game.end_turn()

    result = game.finalize()
    result.final_scores = {policy: result.final_scores[player]
                           for policy, player in zip(agent_names, player_names)}
    result.winner = agent_names[player_names.index(result.winner)]
    log_event("game.finished", winner=result.winner, scores=result.final_scores,
              total_turns=result.total_turns, deck_exhaustions=result.deck_exhaustions)
    return result


def run_batch(
    matchup: Sequence[str] = ("Agent1", "Agent2"), *, games: int = 100,
    seed: int = 1,
) -> BatchSummary:
    """Run a reproducible batch; each game uses consecutive seeds."""
    if games <= 0:
        raise ValueError("games must be positive")
    totals: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    results: list[dict] = []
    turn_total = 0
    for offset in range(games):
        result = play_game(matchup, seed=seed + offset)
        wins[result.winner] += 1
        turn_total += result.total_turns
        totals.update(result.final_scores)
        results.append({"seed": seed + offset, **asdict(result)})
    return BatchSummary(
        matchup=list(matchup), games=games,
        wins={name: wins[name] for name in matchup},
        average_scores={name: totals[name] / games for name in matchup},
        average_turns=turn_total / games, results=results,
    )


def summary_json(summary: BatchSummary) -> str:
    """Serialize a batch summary as readable JSON."""
    return json.dumps(asdict(summary), indent=2, sort_keys=True)
