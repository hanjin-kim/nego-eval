"""A match is several games with the same people in them.

Every negotiation benchmark that lets an agent pick its counterparty throws the
memory away at the final bell, and every one that keeps memory assigns the
counterparty. So nobody has asked the question this file exists for: does an
agent that remembers last game choose differently this game?

The manipulation is one bit.

    carry_over=True    each game starts with the match's ledger in memory
    carry_over=False   each game starts among strangers

Everything else is identical — same seeds, same seller types, same rules, same
number of rounds. Seller health resets between games on purpose: if a seller
arrived at game two already half-bankrupt, the carry-over condition would differ
in the *state* of the world and not only in what the agents know, and the
comparison would no longer isolate memory.

The score is what the buyer earned in the games themselves. Carried rounds are
memory, never income; a match cannot be won by remembering a profit twice.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from nego_eval.game.denom import STEP
from nego_eval.sim.world import Outcome, World


@dataclass
class Ledger:
    """What one party did about the losses it caused, over a whole match."""

    bought: int = 0
    failed: int = 0
    paid: int = 0        # of the loss
    owed: int = 0        # total loss its failures created
    impasses: int = 0

    @property
    def borne(self) -> float:
        """Share of the loss it took on. Undefined before it has failed."""
        return self.paid / self.owed if self.owed else 0.0


def tally(rounds: list[Outcome]) -> dict[str, Ledger]:
    out: dict[str, Ledger] = defaultdict(Ledger)
    for o in rounds:
        L = out[o.seller]
        L.bought += 1
        if o.failed:
            L.failed += 1
            L.paid += o.seller_share
            L.owed += o.loss
            L.impasses += int(o.impasse)
    return dict(out)


@dataclass(frozen=True)
class MatchResult:
    carry_over: bool
    games: tuple[tuple[Outcome, ...], ...]
    profit: int
    per_game: tuple[int, ...]
    ledger: dict[str, Ledger]
    seller_memory: bool = True

    @property
    def rounds(self) -> list[Outcome]:
        return [o for g in self.games for o in g]

    def loyalty(self) -> float:
        """Fraction of volume that went to the single most-used seller."""
        r = self.rounds
        if not r:
            return 0.0
        c: dict[str, int] = defaultdict(int)
        for o in r:
            c[o.seller] += 1
        return max(c.values()) / len(r)

    def switching(self) -> float:
        """Fraction of consecutive rounds that changed seller, across the match."""
        r = self.rounds
        if len(r) < 2:
            return 0.0
        return sum(1 for a, b in zip(r, r[1:]) if a.seller != b.seller) / (len(r) - 1)

    def crossing(self) -> float | None:
        """Of the first rounds of games 2+, how often the buyer stayed put.

        This is the measurement the match exists for. It is the one moment where
        carrying memory can change a choice and nothing else can: the ledger is
        the only thing that distinguishes the conditions, and no within-game
        evidence has accumulated yet.
        """
        if len(self.games) < 2:
            return None
        stays, n = 0, 0
        for prev, cur in zip(self.games, self.games[1:]):
            if not prev or not cur:
                continue
            n += 1
            stays += int(cur[0].seller == prev[-1].seller)
        return stays / n if n else None


@dataclass
class Match:
    """Run `games` games with one buyer against one persistent cast of sellers."""

    buyer_factory: object                 # () -> Buyer, fresh each game
    seller_factory: object                # (game_index) -> list[Seller]
    games: int = 4
    rounds: int = 12
    loss: int = 120
    value: int = 150
    seed: int = 0
    carry_over: bool = True
    #: With carry-over on, whether the sellers see the carried rounds as well.
    #: False isolates the buyer's choice: the ledger reaches the side that picks
    #: and not the side that delivers.
    seller_memory: bool = True
    #: Relabel the carried rounds in the buyer's view. Same volume, same format,
    #: no link to the seller it describes — the control for "used the record"
    #: that withholding the record cannot be.
    scramble: dict[str, str] | None = None
    step: int = STEP
    bargain_rounds: int = 3
    cooloff: int = 3
    _played: list[list[Outcome]] = field(default_factory=list)

    def run(self) -> MatchResult:
        prior: list[Outcome] = []
        games, profits = [], []
        for g in range(self.games):
            w = World(
                buyer=self.buyer_factory(),
                sellers=self.seller_factory(g),
                value=self.value, loss=self.loss, rounds=self.rounds,
                seed=self.seed * 1000 + g,
                bargain_rounds=self.bargain_rounds, cooloff=self.cooloff,
                step=self.step, seller_memory=self.seller_memory,
                scramble=self.scramble,
                prior=list(prior) if self.carry_over else [])
            h = w.run()
            games.append(tuple(h))
            profits.append(sum(o.buyer_profit for o in h))
            if self.carry_over:
                prior = prior + h
        return MatchResult(
            carry_over=self.carry_over, seller_memory=self.seller_memory,
            games=tuple(games),
            profit=sum(profits), per_game=tuple(profits),
            ledger=tally([o for g in games for o in g]))
