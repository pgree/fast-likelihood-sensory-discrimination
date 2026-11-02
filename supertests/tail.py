"""Supertests: the d = 2 network outside its training region ||eta|| <= R = 12.

p_tri evaluates the network at every eta, including ||eta|| > R, where it was not trained
and where h is within 2.2e-5 of 1.  For lambda in the training range
1e-3 <= lambda_2 <= lambda_1 <= 1.499 and eta >= 0:

  network_within_2_2e_5_of_one_beyond_R   1 - hhat <= 2.2e-5 for 12 <= ||eta|| <= 1000
  logit_grows_at_least_linearly           slope(eta) >= ||eta|| for all eta >= 0

Super proves supertests: a pass means the assertion holds for every input that satisfies
assume(), for this Python code evaluated in double precision, not only at sampled inputs;
a failure comes with a counterexample.  The second passed on 2026-11-02.  The first passed
on 2026-10-22 and 2026-11-02 for the same hhat (the same weights and arithmetic, in an
earlier copy of network.py); checked from this file, Super's analysis stopped at its time
limit.

The second covers ||eta|| > 1000, which floats make awkward to check directly.  The logit is
within network.K = 323 of slope(eta) (computed in make_network.py; not checked by Super),
so for ||eta|| >= 1000 the second gives
logit >= 1000 - 323 and 1 - hhat < 1e-294.  Since slope and ||eta|| are positively
homogeneous, checking it where 0 <= eta_i <= 1 and eta_1 + eta_2 >= 1 covers every eta >= 0.  With the first, 1 - hhat <= 2.2e-5 for every ||eta|| >= 12.

Run with Schematic's Super (https://docs.schematic.tech/super):

    super check supertests/tail.py
"""
import math

from schematic import *

from supertests.network import hhat, slope

LAM_LO, LAM_HI = 1e-3, 1.5 - 1e-3          # range of lambda in training
R, ETA_MAX = 12.0, 1000.0


def in_tail(lam1, lam2, eta1, eta2):
    return (LAM_LO <= lam2 <= lam1 <= LAM_HI and eta1 >= 0.0 and eta2 >= 0.0
            and R ** 2 <= eta1 ** 2 + eta2 ** 2 <= ETA_MAX ** 2)


@supertest
def network_within_2_2e_5_of_one_beyond_R(lam1: float, lam2: float, eta1: float, eta2: float):
    assume(in_tail(lam1, lam2, eta1, eta2))
    assert 1.0 - hhat(lam1, lam2, eta1, eta2) <= 2.2e-5


@supertest
def logit_grows_at_least_linearly(eta1: float, eta2: float):
    # Every ray eta >= 0 meets this region and both sides are positively homogeneous, so it
    # covers every eta >= 0.  It also keeps eta away from 0, where the products in slope
    # underflow (at eta = (5e-324, 0) slope returns 0.0 and hypot 5e-324).
    assume(0.0 <= eta1 <= 1.0 and 0.0 <= eta2 <= 1.0 and eta1 + eta2 >= 1.0)
    assert slope(eta1, eta2) >= math.hypot(eta1, eta2)
