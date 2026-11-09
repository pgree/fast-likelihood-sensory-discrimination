"""A reproducible example for Schematic, kept apart from supertests/, which the paper relies on.

The property: for 1e-3 <= lambda_2 <= lambda_1 <= 1.499 and eta >= 0 with ||eta|| >= 12, the
d = 2 network hhat (supertests/network.py) is within 2.2e-5 of 1.  supertests/tail.py proves it
in two parts: 12 <= ||eta|| <= 1000, and linear growth of the logit beyond.  Here it is one
supertest with no upper bound on eta.

The first supertest below is false for this Python code at the extremes of the float range,
although the mathematical property holds:

    eta = (1e103, 0)   z ** 3 in gelu raises OverflowError (float ** raises on overflow)
    eta = (inf, 0)     hhat returns nan

The second is the same statement for hhat with each eta_i clamped at 1000.  The clamp changes no
value hhat returns, since hhat is exactly 1.0 once some eta_i >= 1000, and makes it defined for
every float, so the second supertest is true.

From the repository root:

    python -m schematic_feedback.one_shot_tail      # the failing inputs, in plain Python
    super check schematic_feedback/one_shot_tail.py
"""
import math

from schematic import *

from supertests.network import hhat

CLAMP = 1000.0


def hhat_clamped(lam1, lam2, eta1, eta2):
    return hhat(lam1, lam2, min(eta1, CLAMP), min(eta2, CLAMP))


def in_tail(lam1, lam2, eta1, eta2):
    # eta * eta rather than eta ** 2, which raises OverflowError from about 1.3e154
    return (1e-3 <= lam2 <= lam1 <= 1.499 and eta1 >= 0.0 and eta2 >= 0.0
            and eta1 * eta1 + eta2 * eta2 >= 144.0)


@supertest
def network_within_2_2e_5_of_one_beyond_12(lam1: float, lam2: float, eta1: float, eta2: float):
    assume(in_tail(lam1, lam2, eta1, eta2))
    assert 1.0 - hhat(lam1, lam2, eta1, eta2) <= 2.2e-5


@supertest
def clamped_network_within_2_2e_5_of_one_beyond_12(lam1: float, lam2: float, eta1: float,
                                                   eta2: float):
    assume(in_tail(lam1, lam2, eta1, eta2))
    assert 1.0 - hhat_clamped(lam1, lam2, eta1, eta2) <= 2.2e-5


if __name__ == '__main__':
    lam1, lam2 = 1.0, 0.5
    for eta1 in (12.0, 1e3, 1e103, math.inf):
        for name, f in (('hhat', hhat), ('hhat_clamped', hhat_clamped)):
            try:
                gap = 1.0 - f(lam1, lam2, eta1, 0.0)
                out = f'1 - {name} = {gap!r}, assertion {"holds" if gap <= 2.2e-5 else "fails"}'
            except Exception as e:
                out = f'{name} raises {type(e).__name__}: {e}'
            print(f'lambda = ({lam1}, {lam2}), eta = ({eta1:g}, 0): {out}')
