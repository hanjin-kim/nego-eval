"""Bargaining over a loss, with an outside option that hurts both sides.

The earlier version asked the buyer "how much do you accept" after telling it the
shares must sum to L and what the seller pays. Those two facts leave one
arithmetic answer, so the buyer always produced it — and that looked like a
finding about agents declining to negotiate. It was a finding about the prompt.

A negotiation needs three things this module supplies:

**Rounds.** An offer that cannot be refused is not an offer. Either side may
counter until `max_rounds`.

**A default on impasse.** The loss has already landed on the buyer; if nobody
agrees, it stays there. That asymmetry is what the seller can hold out against,
and what the buyer is trading away when it concedes.

**A cost to impasse that both sides know in advance.** Without it the seller
should never concede: refusing is free. Here an impasse suspends the pair for
`cooloff` rounds — the seller stops quoting to that buyer. The buyer loses an
option, the seller loses a customer, and both are told so before they bargain.
"""

from __future__ import annotations

from dataclasses import dataclass

from nego_eval.game.denom import snap


@dataclass(frozen=True)
class Bargain:
    """What the two sides settled on, and how."""

    loss: int
    seller_share: int
    buyer_share: int
    rounds_used: int
    impasse: bool
    transcript: tuple[str, ...] = ()

    @property
    def settles(self) -> bool:
        return self.seller_share + self.buyer_share == self.loss


TERMS = (
    "If you do not agree within {max_rounds} exchanges, the loss stays where it "
    "landed: the buyer bears all {loss} of it, and the two of you stop trading "
    "for the next {cooloff} rounds. Both of you lose from that."
)


def negotiate(loss: int, seller, buyer, seller_name: str, history,
              max_rounds: int = 3, cooloff: int = 3, remaining: int = 10,
              step: int = 1, seller_history=None) -> Bargain:
    """Alternating offers over the seller's share. Seller opens.

    Every amount is snapped to `step` on the way in, so the legal moves are the
    same finite set for a table, a script and a model alike. Snapping here rather
    than inside each agent means an agent cannot leave the grid by ignoring it.
    """
    terms = TERMS.format(max_rounds=max_rounds, loss=loss, cooloff=cooloff)
    log: list[str] = []
    #: Normally the same rounds. They come apart in the condition that gives the
    #: buyer a carried ledger and the seller none, which turns the split between
    #: "the buyer chose better" and "the seller tried harder" from a subtraction
    #: into a design.
    sh = history if seller_history is None else seller_history
    offer = snap(seller.open_offer(loss, sh), step, hi=loss)
    log.append(f"seller offers to pay {offer}")

    for r in range(1, max_rounds + 1):
        accept, ask = buyer.bargain(loss, offer, r, max_rounds, seller_name,
                                    history, terms)
        if accept:
            log.append(f"buyer accepts (round {r})")
            return Bargain(loss, offer, loss - offer, r, False, tuple(log))
        ask = snap(int(ask), step, hi=loss)
        log.append(f"buyer asks seller to pay {ask}")
        if r == max_rounds:
            break
        new = snap(seller.respond(loss, ask, r, sh,
                                  remaining=remaining, cooloff=cooloff), step, hi=loss)
        log.append(f"seller counters {new}")
        if new >= ask:                                 # seller met or beat the ask
            return Bargain(loss, new, loss - new, r + 1, False, tuple(log))
        offer = new

    log.append("impasse")
    return Bargain(loss, 0, loss, max_rounds, True, tuple(log))
