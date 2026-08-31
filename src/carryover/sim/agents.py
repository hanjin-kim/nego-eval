"""Sellers that differ in one thing only, and buyers that may or may not notice.

The three sellers share a reliability. That is deliberate. If one seller failed
less often, a buyer returning to it would tell us only that the buyer can count
failures — a much weaker claim than the one the experiment is about. Here the
only difference is what happens *after* a failure: who offers to eat the loss,
and whether a promise made in round t is kept in round t+k.

So a buyer that ends up trading repeatedly with the cooperative seller has
learned something about conduct under conflict, which is the thing worth
measuring.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field

from carryover.sim.world import Outcome, Quote


@dataclass
class ScriptedSeller:
    """A fixed bargaining type whose *care* depends on who is buying.

    The first version held reliability equal across sellers so that the only
    difference was conduct after a failure. That looked like a clean design and
    was the wrong one: the empirical literature on relational contracting finds
    the opposite arrangement. Relational buyers pay more, and the rent is what
    disciplines a supplier into the things a contract cannot specify — being
    reliable, holding spare capacity for you. Macchiavello and others measure
    suppliers earning higher margins on relational orders precisely because those
    rents buy effort.

    So reliability is not a constant here. It is bought:

        reliability = base + bonus x loyalty

    where loyalty is this buyer's recent share of business. Switching every round
    keeps every seller at `base`; staying lifts the one you stay with. That makes
    the return on a relationship compound — fewer failures, which makes the
    relationship worth more — instead of sitting in a single post-failure
    concession that the opening offer already dominated.
    """

    name: str
    base_price: int
    reliability: float = 0.70          # what a stranger gets
    care_bonus: float = 0.25           # what loyalty can add
    loyalty_window: int = 8
    share: float = 0.5              # opening offer, as a fraction of the loss
    floor: float = 0.5              # most it will ever concede to
    keeps_promises: bool = True
    promise_rate: float = 0.0       # how often it offers future compensation
    cost: int = 70                  # its own unit cost; margin drives concessions
    jitter: int = 12                # exogenous per-round price noise
    #: Buyer-power regime. The earlier design tied a seller's type to its list
    #: price, which let a buyer read the answer off the quotes without ever
    #: trading — a shortcut, and one with no basis here since the sellers share a
    #: capacity and a baseline reliability. The empirical literature has it both
    #: ways: relational buyers pay a premium where supply is scarce, and take a
    #: discount where they hold the power. This is the second regime. Every seller
    #: opens at the same price and only a returning buyer is quoted less, so the
    #: types are invisible until someone stays long enough to find out.
    loyalty_discount: int = 0
    #: A seller squeezed below its costs does not keep performing. Until now these
    #: sellers conceded and discounted without limit, which quietly removed the
    #: reason relational contracts exist at all: a buyer needs its supplier to
    #: survive and to keep investing. Here the running margin feeds back into
    #: reliability, so the cheapest quote is not free — press hard enough and the
    #: thing you are buying gets worse, then stops.
    #: How much the floor rises with this buyer's recent share of the business.
    #: Without it a relationship is only informative — you learn a fixed type and
    #: then remembering is worth nothing but the time it saves. The empirical
    #: literature has it the other way round: the rent a returning buyer pays is
    #: what buys effort a contract cannot specify, so staying has to *change* the
    #: counterparty and not merely reveal it. How much it changes is itself a
    #: hidden trait, found out by staying.
    floor_care: float = 0.0
    floor_cap: float = 0.80
    solvency_floor: int = -150      # cumulative margin at which it fails outright
    health_scale: int = 300         # margin over which reliability recovers fully
    #: Whether running margin feeds back into delivery. It normally should — a
    #: squeezed supplier cuts corners — but note the ramp is two-sided: a seller
    #: with no trades yet sits at a third of full health and therefore delivers
    #: below its nominal rate from the first round. On a board that publishes the
    #: rate and asks the agent to price it, that gap makes the published number a
    #: lie and puts the estimation noise back. Switch it off there, and say so.
    health_couples: bool = True
    _margin: int = field(default=0, repr=False)

    def book(self, delta: int) -> None:
        self._margin += delta

    def health(self) -> float:
        """0 at the solvency floor, 1 once comfortably in profit."""
        if self._margin <= self.solvency_floor:
            return 0.0
        return min(1.0, (self._margin - self.solvency_floor) /
                   (self.health_scale - self.solvency_floor))

    def alive(self) -> bool:
        return self._margin > self.solvency_floor
    seed: int = 0
    _owed: int = field(default=0, repr=False)
    _rng: random.Random = field(default=None, repr=False)

    def quote(self, t: int, remaining: int) -> int:
        # Prices move. Without that, "always take the cheapest" is a constant
        # choice and persistence measures price stability rather than loyalty —
        # the metric would read 1.00 for a policy with no memory at all.
        # The jitter is exogenous and independent of conduct, which is also what
        # makes the trust premium identifiable: a relational buyer must sometimes
        # pay more, on purpose, to be distinguishable from a cheap one.
        if self._rng is None:
            # Not `hash`. Python randomises string hashing per process, so the
            # same seed produced different prices in different runs and the
            # environment was only reproducible inside one process — which the
            # reproducibility test, running in one process, could not see. A
            # benchmark whose board changes between runs is not a benchmark.
            self._rng = random.Random(
                zlib.crc32(f"{self.name}:{self.seed}".encode()) & 0xFFFFFFFF)
        d = self._rng.randint(-self.jitter, self.jitter)
        d -= int(round(self.loyalty_discount * self._loyalty))
        if self._owed and self.keeps_promises:
            d -= self._owed
            self._owed = 0
        return max(1, self.base_price + d)

    _loyalty: float = field(default=0.0, repr=False)

    def note_history(self, history: list[Outcome]) -> None:
        self._loyalty = self.loyalty_of(history)

    def loyalty_of(self, history: list[Outcome]) -> float:
        """Share of the last `loyalty_window` rounds that came here.

        The denominator is the window, not however many rounds happen to exist.
        Dividing by the rounds played makes a relationship out of a coincidence:
        the seller picked in round one sees a window of length one containing
        itself and reads full loyalty, so every fresh start hands whoever goes
        first the standing of an established partner. It also made the number a
        function of how much history the seller was shown, which is the variable
        under test.

        Against a fixed denominator a relationship has to accumulate, which is
        also the shape the empirical work finds — beliefs rising in the count of
        past shipments rather than in their share.
        """
        w = history[-self.loyalty_window:]
        return sum(1 for o in w if o.seller == self.name) / self.loyalty_window

    def effective_reliability(self, history: list[Outcome]) -> float:
        self.note_history(history)
        base = min(0.99, self.reliability + self.care_bonus * self._loyalty)
        if not self.health_couples:
            return base
        # A seller running at a loss cuts corners before it disappears.
        return base * (0.55 + 0.45 * self.health())

    def open_offer(self, loss: int, history: list[Outcome]) -> int:
        return int(round(loss * self.share))

    def respond(self, loss: int, ask: int, r: int, history: list[Outcome],
                remaining: int = 10, cooloff: int = 3) -> int:
        """Concede up to what the relationship is worth to *this* seller.

        The earlier version compared the ask to a constant. That made the impasse
        threat one-sided: the prompt told both parties an impasse costs them, but
        only the buyer's policy responded to it. A seller with a fixed floor is
        indifferent to losing the customer, so "we both lose" was not true of the
        environment — only of the text.

        Here the ceiling on a concession is the margin the seller expects to give
        up during the suspension:

            worth = margin x min(cooloff, remaining) x loyalty

        `loyalty` is how much of the buyer's business this seller has been
        getting — a seller nobody returns to has little to protect and holds out.
        The floor still caps generosity (that is the seller's type), but near the
        end of the run the relationship is worth less and every type hardens,
        which is the endgame effect worth being able to observe.

        It is the same window `note_history` uses, and it has to be. Taken over
        all of history instead, the number would say something different
        depending on how much history there is: concentrating on one seller
        during a twelve-round game reads as 0.9 when the game is all there is and
        as 0.6 when four games are, so a seller would concede *less* precisely
        because it remembers more. That makes the parameter a function of the
        manipulated variable, which is the one thing it must not be.
        """
        margin = max(self.base_price - self.cost, 0)
        loyalty = self.loyalty_of(history)
        # A stranger still has something to protect: the sale it is in the middle
        # of, and the chance of another. Multiplying straight by loyalty made that
        # zero, and since loyalty counts only the last `loyalty_window` rounds, a
        # seller met for the first time could concede nothing at all — so pushing
        # back moved it *down* from its opening offer and the only rational reply
        # to any offer was to take it. That is not a bargaining environment.
        #
        # The relationship still raises the ceiling, which is the point; it no
        # longer decides whether there is one.
        standing = 0.4 + 0.6 * loyalty
        worth = margin * min(cooloff, max(remaining, 0)) * standing
        earned = min(self.floor_cap, self.floor + self.floor_care * loyalty)
        ceiling = min(int(round(loss * earned)), int(worth))
        return min(ceiling, ask)

    def earned_floor(self, history: list[Outcome]) -> float:
        """What this seller would concede to, given the relationship so far."""
        return min(self.floor_cap,
                   self.floor + self.floor_care * self.loyalty_of(history))


@dataclass
class CheapestBuyer:
    """T1 — price only. The null: no memory, no relationship, no conduct."""

    def choose(self, quotes: list[Quote], t: int, remaining: int,
               history: list[Outcome]) -> str:
        return min(quotes, key=lambda q: q.price).seller

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        return True, offer            # takes whatever is offered


@dataclass
class GrudgeBuyer:
    """T2 — a deterministic relational policy, for calibration.

    Not a model of anything. It exists so that persistence has a known ceiling in
    this environment: if an LLM buyer scores below this, it is not using history.
    """

    penalty: int = 60               # price-equivalent cost of a bad loss split

    def _reputation(self, history: list[Outcome]) -> dict[str, float]:
        acc: dict[str, list[int]] = {}
        for o in history:
            if o.failed:
                acc.setdefault(o.seller, []).append(o.seller_share)
        return {k: sum(v) / (len(v) * max(o.loss, 1)) if v else 0.0
                for k, v in acc.items()
                for o in [next(x for x in history if x.seller == k and x.failed)]}

    def choose(self, quotes: list[Quote], t: int, remaining: int,
               history: list[Outcome]) -> str:
        rep = self._reputation(history)
        def score(q: Quote) -> float:
            return q.price - self.penalty * rep.get(q.seller, 0.5)
        return min(quotes, key=score).seller

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        """Ask for half, once. Concede on the last exchange rather than impasse."""
        want = loss // 2
        if offer >= want or r == max_rounds:
            return True, offer
        return False, want


@dataclass
class LLMBuyer:
    """T3 — the model chooses, given the same facts the scripted policies get.

    The prompt states an objective and the ledger. It does not contain the words
    trust, relationship, loyalty, partner, or reputation. Those words would make
    the experiment a compliance test: an agent told to value relationships will
    value relationships, and the resulting persistence would say nothing about
    whether the payoffs justify it.

    What the model is told instead: the loss is real, it is split, and rounds
    remain. Whether that is enough to produce loyalty is the question.
    """

    model: str | None = None
    log: list[dict] = field(default_factory=list)
    #: Calls that failed after every retry. The buyer falls back to the cheapest
    #: quote rather than aborting the match, because losing ninety-nine good
    #: calls to one bad one is worse than one degraded round — but a run with a
    #: material number of these is not a measurement of the model, so the count
    #: travels with the result.
    fallbacks: int = 0
    #: Output tokens this buyer spent, reasoning included. Every model faces the
    #: same seventy decisions a match — the rules fix that — so the only thing
    #: that varies is how long it thinks about each, and one model has spent
    #: twenty-five times another's wall clock for the same seventy choices. Left
    #: unmeasured, a score rewards deliberation the buyer would be paying for in
    #: any real procurement. Tokens rather than seconds: seconds move with
    #: provider load and with whatever else this machine is running.
    tokens: int = 0

    SYSTEM = (
        "You are a purchasing agent. Your only objective is to maximise your own "
        "cumulative profit over the whole run.\n"
        "Each round you buy one unit. If delivery succeeds you gain (value - price). "
        "If it fails you pay the price and a share of a loss; the seller pays the rest. "
        "The two shares always sum to the full loss — it does not disappear.\n"
        "Sellers do not publish their failure rates. Past rounds are the only "
        "evidence you have about how a seller behaves when a delivery fails.\n"
        "Answer with JSON only."
    )

    def _ask(self, user: str) -> dict:
        """One call, with the output tokens it cost charged to this buyer.

        Read as a difference across the process-wide tally, which is only sound
        while one request per process is in flight — the runner holds a
        semaphore of one for exactly that reason. Two concurrent calls on the
        same model would charge each other's tokens to whichever finished last.
        """
        from carryover.sim import llm as _llm
        before = dict(_llm.per_model.get(self.model, {}))
        try:
            return _llm.ask_json(self.SYSTEM, user, model=self.model)
        finally:
            after = _llm.per_model.get(self.model, {})
            self.tokens += ((after.get('out', 0) - before.get('out', 0))
                            + (after.get('reasoning', 0) - before.get('reasoning', 0)))

    def _ledger(self, history: list[Outcome]) -> str:
        if not history:
            return "No rounds yet."
        agg: dict[str, dict] = {}
        for o in history:
            a = agg.setdefault(o.seller, {"n": 0, "fail": 0, "paid": 0, "loss": 0,
                                          "promised": 0})
            a["n"] += 1
            if o.failed:
                a["fail"] += 1
                a["paid"] += o.seller_share
                a["loss"] += o.loss
                if o.promise:
                    a["promised"] += 1
        rows = []
        for k, a in sorted(agg.items()):
            share = f"{a['paid']}/{a['loss']}" if a["loss"] else "-"
            rows.append(f"{k}: bought {a['n']}, failed {a['fail']}, "
                        f"seller paid {share} of losses, promises made {a['promised']}")
        last = history[-1]
        rows.append(f"last round: {last.seller}, "
                    f"{'FAILED' if last.failed else 'ok'}, your profit {last.buyer_profit}")
        return "\n".join(rows)

    def choose(self, quotes: list[Quote], t: int, remaining: int,
               history: list[Outcome]) -> str:
        from carryover.sim.llm import ask_json
        q = ", ".join(
            f"{x.seller}={x.price}"
            + (f" (published delivery rate {x.rate:.2f})" if x.rate is not None else "")
            for x in quotes)
        user = (f"Round {t + 1}. Rounds remaining after this one: {remaining - 1}.\n"
                f"Unit value to you if delivery succeeds: 150.\n"
                f"Loss if delivery fails: {history[0].loss if history else 'unknown'}.\n"
                f"Quotes: {q}\n\n"
                f"Your record so far:\n{self._ledger(history)}\n\n"
                f'Reply: {{"seller": "<name>", "why": "<one short clause>"}}')
        try:
            d = self._ask(user)
        except Exception:
            self.fallbacks += 1
            return min(quotes, key=lambda x: x.price).seller
        self.log.append({"t": t, "kind": "choose", "raw": d})
        pick = str(d.get("seller", "")).strip()
        names = {x.seller for x in quotes}
        return pick if pick in names else min(quotes, key=lambda x: x.price).seller

    def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
        from carryover.sim.llm import ask_json
        user = (f"Delivery from {seller_name} failed. The loss is {loss}.\n"
                f"Exchange {r} of {max_rounds}. {seller_name} currently offers to pay {offer}; "
                f"the remaining {loss - offer} would fall on you.\n\n"
                f"{terms}\n\n"
                f"Your record:\n{self._ledger(history)}\n\n"
                f'Either accept, or say what you want {seller_name} to pay instead. '
                f'Reply: {{"accept": true|false, "ask": <integer 0-{loss}>}}')
        try:
            d = self._ask(user)
        except Exception:
            self.fallbacks += 1
            return True, offer          # take the standing offer rather than abort
        self.log.append({"t": len(history), "kind": "bargain", "r": r, "raw": d})
        acc = d.get("accept")
        try:
            ask = int(d.get("ask", offer))
        except (TypeError, ValueError):
            ask = offer
        if acc is None:
            acc = ask <= offer
        return bool(acc), ask
