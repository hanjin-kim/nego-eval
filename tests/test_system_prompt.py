"""The buyer's instructions, checked against the rules they describe.

Two copies of this text existed and they drifted. One told models a failed
delivery costs them the price, which `World` does not charge, and that sellers
do not publish their failure rates, one message before printing all three.
Neither is the sort of thing a test suite normally guards, and that is exactly
why it survived: prose is not executed, so nothing complained.
"""
from nego_eval.rl.vf_env import EVAL, SYSTEM, Driver
from nego_eval.sim.agents import SYSTEM as AGENTS_SYSTEM
from nego_eval.sim.agents import LLMBuyer, ScriptedSeller
from nego_eval.sim.world import World


def test_there_is_one_system_prompt():
    assert SYSTEM is AGENTS_SYSTEM is LLMBuyer.SYSTEM


def test_it_does_not_charge_the_buyer_for_goods_it_never_received():
    """`world.py` sets buyer_profit = -buyer_share on failure, and no price."""
    assert 'pay a share of a loss' in SYSTEM
    assert 'pay the price and a share' not in SYSTEM


def test_a_failed_round_really_does_cost_only_the_share():
    """The rule the sentence above describes, exercised rather than asserted.

    A real `ScriptedSeller` at zero reliability rather than a stand-in, so the
    round goes through the same bargaining the models face.
    """
    class Taker:
        def choose(self, quotes, t, remaining, history):
            return quotes[0].seller

        def bargain(self, loss, offer, r, max_rounds, seller_name, history, terms):
            return True, offer

    seller = ScriptedSeller(name='A', base_price=100, reliability=0.0,
                            care_bonus=0.0, share=0.4, floor=0.4)
    h = World(buyer=Taker(), sellers=[seller], value=150, loss=110,
              rounds=1, seed=1).run()
    assert h[0].failed
    assert h[0].buyer_profit == -h[0].buyer_share
    assert h[0].buyer_profit != -(h[0].price + h[0].buyer_share)


def test_it_does_not_deny_what_the_quote_sheet_prints():
    assert 'do not publish their failure rates' not in SYSTEM
    assert 'publish a delivery rate' in SYSTEM


def test_the_quote_sheet_really_does_print_the_rate():
    text = Driver(900_000, **EVAL).pending.text
    assert 'published delivery rate' in text


def test_what_is_withheld_is_the_loss_split():
    """The one thing the record is for, and the only thing kept off the sheet."""
    assert 'how much of a loss they absorb' in SYSTEM
    text = Driver(900_000, **EVAL).pending.text
    assert 'absorb' not in text
