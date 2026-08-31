"""The environment, and the one thing it will not let an agent talk its way out of.

A round is a trade that may fail. When it fails a loss `L` exists, and the world
settles only when the two shares add up to it:

    L_B + L_S = L

Everything else here — quotes, choice, negotiation, memory — is arrangement.
That identity is the experiment. Capability makes search, phrasing and
re-negotiation cheap; it does not make the loss smaller. If persistent
relationships appear, they appear because someone kept eating a share of a real
number, and this module is where that number is kept honest.

The world never reads an agent's reasoning. It reads the two integers they
settle on, and refuses the round if they do not close.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from carryover.game.denom import snap
from carryover.sim.bargain import Bargain, negotiate


class SettlementError(RuntimeError):
    """Proposed shares do not exhaust the loss."""


@dataclass(frozen=True)
class Quote:
    seller: str
    price: int
    #: The delivery rate the seller publishes, when the board has one. It is
    #: quoted, not estimated, so an agent never has to infer reliability from a
    #: handful of trades — that inference is what made an earlier board's answer
    #: unrecoverable. What stays hidden is conduct after a failure.
    rate: float | None = None


@dataclass(frozen=True)
class Outcome:
    """One completed round, from the world's point of view."""

    t: int
    seller: str
    price: int
    failed: bool
    loss: int
    buyer_share: int
    seller_share: int
    buyer_profit: int
    seller_profit: int
    promise: str | None = None
    impasse: bool = False
    bargain_rounds: int = 0
    transcript: tuple[str, ...] = ()

    def settles(self) -> bool:
        return self.buyer_share + self.seller_share == self.loss


class Seller(Protocol):
    name: str
    reliability: float

    def quote(self, t: int, remaining: int) -> int: ...
    def allocate(self, loss: int, history: list[Outcome]) -> tuple[int, str | None]:
        """Return (share the seller accepts, optional promise). World checks the rest."""
        ...


class Buyer(Protocol):
    def choose(self, quotes: list[Quote], t: int, remaining: int,
               history: list[Outcome]) -> str: ...
    def counter(self, loss: int, seller_share: int, seller: str,
                history: list[Outcome]) -> int:
        """Return the share the buyer accepts. Sum is enforced by the world."""
        ...


@dataclass
class World:
    """Repeated bilateral trade with exogenous failure and endogenous partner choice.

    `value` is what a successful delivery is worth to the buyer; `loss` is what a
    failure destroys. The ratio `loss / price` is the sweep variable of the first
    experiment: at zero there is no economic reason to prefer anyone, and any
    relationship that still forms is a property of the prompt rather than of the
    payoffs.
    """

    buyer: Buyer
    sellers: list[Seller]
    value: int = 150
    loss: int = 80
    rounds: int = 20
    seed: int = 0
    bargain_rounds: int = 3
    cooloff: int = 3
    #: What a failed delivery costs the seller regardless of the split — wasted
    #: handling, a shipment prepared and not completed. Without it a seller is
    #: indifferent to failing, which is not a market.
    fail_cost: int = 20
    #: Rounds carried in from earlier games in the same match. Agents read them
    #: as memory; the world does not settle them again and they never enter a
    #: score. This is the whole manipulated variable of the board game: with
    #: `prior` empty every game starts among strangers, and with it a seller
    #: arrives already carrying what it did last time.
    prior: list[Outcome] = field(default_factory=list)
    #: Denomination the round's money is snapped to. 1 leaves the continuous
    #: behaviour the sim was measured on; the game sets it to `denom.STEP`.
    step: int = 1
    #: Whether the sellers get the carried rounds too. Off, the buyer remembers
    #: the match and the sellers remember only this game, so a seller cannot put
    #: extra care into a relationship it does not know it has.
    seller_memory: bool = True
    #: Permutation applied to the seller names in the buyer's view of the carried
    #: rounds, and to nothing else. Simply withholding the ledger removes
    #: information as well as validity, so a score that falls under it is equally
    #: explained by "had less data" — which is not the claim. Relabelling keeps
    #: the prompt the same length, the same shape and the same numbers, and cuts
    #: only the link between a record and the seller it belongs to. The sellers
    #: still see the truth, so the real return on staying is unchanged and what
    #: is isolated is the buyer's use of the record.
    scramble: dict[str, str] | None = None
    history: list[Outcome] = field(default_factory=list)

    @property
    def memory(self) -> list[Outcome]:
        """What an agent is allowed to remember: the match so far, not the game."""
        return self.prior + self.history

    @property
    def buyer_memory(self) -> list[Outcome]:
        """The same, with the carried names relabelled if a scramble is set."""
        if not self.scramble:
            return self.memory
        import dataclasses
        moved = [dataclasses.replace(o, seller=self.scramble.get(o.seller, o.seller))
                 for o in self.prior]
        return moved + self.history

    def run(self) -> list[Outcome]:
        rng = random.Random(self.seed)
        by_name = {s.name: s for s in self.sellers}
        blocked: dict[str, int] = {}          # seller -> round it becomes available
        for t in range(self.rounds):
            remaining = self.rounds - t
            mem = self.buyer_memory
            smem = (self.memory if self.seller_memory else self.history)
            for s in self.sellers:                 # loyalty is read before quoting
                if hasattr(s, "note_history"):
                    s.note_history(smem)
            quotes = [Quote(s.name, snap(s.quote(t, remaining), self.step),
                            getattr(s, "reliability", None))
                      for s in self.sellers
                      if blocked.get(s.name, -1) <= t
                      and (not hasattr(s, "alive") or s.alive())]
            if not quotes:                    # everyone suspended: no trade this round
                continue
            available = {q.seller for q in quotes}
            pick = self.buyer.choose(quotes, t, remaining, mem)
            if pick not in available:
                # Naming a suspended seller is not an exit either. The cheapest
                # available quote stands in, and the attempt is not recorded as a
                # choice of that seller — the suspension is the point.
                pick = min(quotes, key=lambda q: q.price).seller
            seller = by_name[pick]
            price = next(q.price for q in quotes if q.seller == pick)

            # Reliability is a property of the pair, not of the seller alone.
            rel = (seller.effective_reliability(smem)
                   if hasattr(seller, "effective_reliability") else seller.reliability)
            failed = rng.random() >= rel
            if not failed:
                self.history.append(Outcome(
                    t=t, seller=pick, price=price, failed=False, loss=0,
                    buyer_share=0, seller_share=0,
                    buyer_profit=self.value - price, seller_profit=price - seller.cost))
                if hasattr(seller, "book"):
                    seller.book(price - seller.cost)
                continue

            b = negotiate(self.loss, seller, self.buyer, pick, mem,
                          max_rounds=self.bargain_rounds, cooloff=self.cooloff,
                          remaining=remaining, step=self.step,
                          seller_history=smem)
            if b.impasse:
                # The pair stops trading. Both lose: the buyer an option, the
                # seller a customer. Both were told this before bargaining.
                blocked[pick] = t + self.cooloff

            self.history.append(Outcome(
                t=t, seller=pick, price=price, failed=True, loss=self.loss,
                buyer_share=b.buyer_share, seller_share=b.seller_share,
                # No delivery, no price. The buyer does not pay for goods it never
                # received, and the seller does not collect for them — that is the
                # default in any sale-of-goods regime, and getting it wrong made
                # failure *profitable* for a seller that concedes nothing, which
                # would invert the incentives the moment a seller is a model
                # rather than a script.
                #
                # What survives the failure is the consequential loss, and that is
                # what the two of them bargain over.
                buyer_profit=-b.buyer_share,
                seller_profit=-b.seller_share - self.fail_cost,
                impasse=b.impasse, bargain_rounds=b.rounds_used,
                transcript=b.transcript))
            if hasattr(seller, "book"):
                seller.book(-b.seller_share - self.fail_cost)
        return self.history
