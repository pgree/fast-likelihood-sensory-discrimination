"""p_duo for the duo-trio test with equal covariances (Section 4).

With Sigma_x = Sigma_y = Sigma, p_duo depends on the stimuli only through
r = (mu_y - mu_x)^T Sigma^{-1} (mu_y - mu_x), for any d (Claim 3):

    p_duo(r) = P( chi'^2_d(lambda_1) > chi'^2_d(lambda_0) ),   lambda_{1,0} = r (2 +- sqrt 3) / 6
             = int_0^inf Fbar_{d, lambda_1}(t) f_{d, lambda_0}(t) dt

with f and Fbar the density and the upper tail of the noncentral chi-squared.
The integral is evaluated by adaptive Gauss-Kronrod quadrature (QUADPACK,
through scipy) and p_duo is stored as a Chebyshev expansion on [0, R]
(Section 4.1).  The tolerance eps is split three ways: R is where 1 - p_duo
falls below eps/10, the node values are computed by quadrature at eps/10, and
n is large enough that the coefficients dropped sum to less than eps/10.
"""
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.stats import ncx2

warnings.simplefilter('ignore', IntegrationWarning)   # the reference runs ask more than quad can certify
SQRT3 = np.sqrt(3.0)
QUAD_FLOOR = 1e-14          # the tightest tolerance quad can meet in double precision
SPLIT = 10.0                # each error term gets eps / SPLIT


def lambdas(r):
    return r * (2.0 + SQRT3) / 6.0, r * (2.0 - SQRT3) / 6.0


def pduo_quad(r, d, eps=1e-10):
    """p_duo(r) by adaptive quadrature."""
    lam_1, lam_0 = lambdas(float(r))
    val, _ = quad(lambda t: ncx2.sf(t, d, lam_1) * ncx2.pdf(t, d, lam_0),
                  0.0, np.inf, epsabs=eps, epsrel=eps, limit=400)
    return float(val)


def one_minus_pduo_quad(r, d, eps=1e-10):
    """1 - p_duo(r) as its own integral, which keeps relative accuracy at large r."""
    lam_1, lam_0 = lambdas(float(r))
    val, _ = quad(lambda t: ncx2.sf(t, d, lam_0) * ncx2.pdf(t, d, lam_1),
                  0.0, np.inf, epsabs=1e-15, epsrel=eps, limit=400)
    return float(val)


def truncation_radius(d, eps):
    """The smallest grid point R at which 1 - p_duo(R) < eps, by doubling then bisection
    to a relative width of 1e-3; the upper end of the final bracket is returned."""
    hi = 1.0
    while one_minus_pduo_quad(hi, d) > eps:
        hi *= 2.0
    lo = 0.0 if hi == 1.0 else hi / 2.0
    while hi - lo > 1e-3 * hi:
        mid = 0.5 * (lo + hi)
        if one_minus_pduo_quad(mid, d) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def chebyshev_nodes(n, r_max):
    """The n Chebyshev nodes of the first kind, mapped to [0, r_max]."""
    return 0.5 * r_max * (np.cos(np.pi * (np.arange(n) + 0.5) / n) + 1.0)


def chebyshev_coefficients(values):
    """Coefficients in the T basis of the degree n-1 interpolant through the node values."""
    n = len(values)
    k = np.arange(n)
    c = (2.0 / n) * (values * np.cos(np.pi * k[:, None] * (k + 0.5) / n)).sum(axis=1)
    c[0] *= 0.5
    return c


def evaluate(r, coef, r_max):
    """The expansion at r by Clenshaw recurrence; r beyond r_max is clamped (p_duo is 1 there)."""
    x = np.clip(2.0 * np.asarray(r, dtype=float) / r_max - 1.0, -1.0, 1.0)
    b1 = np.zeros_like(x)
    b2 = np.zeros_like(x)
    for c in coef[:0:-1]:
        b1, b2 = 2.0 * x * b1 - b2 + c, b1
    return x * b1 - b2 + coef[0]


def expansion(d, eps=1e-8, n=36):
    """The paper's expansion of p_duo for dimension d, without the checks of build:
    (coefficients, r_max).  About a second; build once and reuse."""
    share = eps / SPLIT
    r_max = truncation_radius(d, share)
    values = np.array([pduo_quad(v, d, max(share, QUAD_FLOOR)) for v in chebyshev_nodes(n, r_max)])
    return chebyshev_coefficients(values), r_max


# The paper's expansions, eps = 1e-8 and 36 nodes: d -> (r_max, coefficients in the T basis
# on [0, r_max]).  expansion(d) rebuilds them; the rebuild agrees to 2e-14 in the coefficients.
TABLE = {
    2: (218.375, np.array([
         9.46360262561416588e-01,
         1.02987993636222241e-01,
        -9.12577310816688470e-02,
         7.49753667833056997e-02,
        -5.75144564852512197e-02,
         4.15612296807339812e-02,
        -2.85702303864401666e-02,
         1.88658066915490391e-02,
        -1.20693269721539927e-02,
         7.52922039992166923e-03,
        -4.59846930927393979e-03,
         2.75424885724300699e-03,
        -1.61770862122433590e-03,
         9.30798253580864248e-04,
        -5.23917955197455358e-04,
         2.88096787396108223e-04,
        -1.54597481316508402e-04,
         8.08922320132348613e-05,
        -4.12500179013952639e-05,
         2.04939896559785950e-05,
        -9.91888928643534901e-06,
         4.67670939866610676e-06,
        -2.14836243590039205e-06,
         9.61710966051403899e-07,
        -4.19618050171186327e-07,
         1.78507164309375327e-07,
        -7.40594420326849139e-08,
         2.99756532620401260e-08,
        -1.18404026321227019e-08,
         4.56588028188400679e-09,
        -1.71947209248735078e-09,
         6.32598936197344535e-10,
        -2.27426337793021655e-10,
         7.98568317953810842e-11,
        -2.71758040453700598e-11,
         8.26341714961949602e-12,
    ])),
    3: (220.75, np.array([
         9.43382092851809628e-01,
         1.08391854011613512e-01,
        -9.51882376742339137e-02,
         7.69719706214331417e-02,
        -5.76399514919981593e-02,
         4.02633655728838746e-02,
        -2.64521406990986946e-02,
         1.64839904679326514e-02,
        -9.82185771358829764e-03,
         5.63384753714252836e-03,
        -3.12649946483134489e-03,
         1.68338533897359000e-03,
        -8.79979125805846994e-04,
         4.46033275427575797e-04,
        -2.18554739191014369e-04,
         1.03037861072435843e-04,
        -4.64153860021379293e-05,
         1.97667073416032622e-05,
        -7.81524950672444521e-06,
         2.76653252947086249e-06,
        -7.97771432908205702e-07,
         1.18147469233923858e-07,
         6.66263946606330541e-08,
        -8.55081462056459286e-08,
         6.25128862933680700e-08,
        -3.78040435267740807e-08,
         2.05849076950759269e-08,
        -1.04372165509057791e-08,
         5.01178373030825600e-09,
        -2.30167960154255715e-09,
         1.01738658701370578e-09,
        -4.34710263156704915e-10,
         1.80084098096629676e-10,
        -7.23875280704709829e-11,
         2.79453879018879763e-11,
        -9.41153385934799697e-12,
    ])),
}


def pduo(r, d):
    """p_duo(r) from the stored expansion, d = 2 or 3.  r beyond r_max is clamped, where
    p_duo is within 1e-9 of 1."""
    r_max, coef = TABLE[d]
    return evaluate(r, coef, r_max)


def build(d, eps, n):
    """The expansion for dimension d at tolerance eps with n nodes, and its error terms:
    quad_err  the node values against quadrature at a 100x tighter tolerance
    tail      twice the sum of the coefficients beyond n in a build with 2n nodes
    sampled   the largest deviation from quadrature between the nodes"""
    share = eps / SPLIT
    quad_eps = max(share, QUAD_FLOOR)
    ref_eps = max(quad_eps / 100.0, QUAD_FLOOR)
    r_max = truncation_radius(d, share)
    nodes = chebyshev_nodes(n, r_max)
    values = np.array([pduo_quad(v, d, quad_eps) for v in nodes])
    coef = chebyshev_coefficients(values)
    fine = chebyshev_coefficients(np.array([pduo_quad(v, d, quad_eps) for v in chebyshev_nodes(2 * n, r_max)]))
    between = chebyshev_nodes(4 * n + 1, r_max)
    ref = np.array([pduo_quad(v, d, ref_eps) for v in between])
    tight = np.array([pduo_quad(v, d, ref_eps) for v in nodes])
    return dict(r_max=r_max, coef=coef, share=share,
                quad_err=float(np.max(np.abs(values - tight))),
                tail=float(2.0 * np.sum(np.abs(fine[n:]))),
                sampled=float(np.max(np.abs(evaluate(between, coef, r_max) - ref))))


def doubling(d, coef, r_max, eps=1e-8, n_grid=100_001):
    """Sup over [0, r_max] of the difference between the expansion and the one built
    the same way from twice as many nodes: (difference, r at which it occurs)."""
    n = len(coef)
    fine = chebyshev_coefficients(np.array([pduo_quad(v, d, max(eps / SPLIT, QUAD_FLOOR))
                                            for v in chebyshev_nodes(2 * n, r_max)]))
    r = np.linspace(0.0, r_max, n_grid)
    diff = np.abs(evaluate(r, coef, r_max) - evaluate(r, fine, r_max))
    k = int(np.argmax(diff))
    return float(diff[k]), float(r[k])


def series_terms(d, r, eps=1e-8):
    """Terms of the series of Ennis and Johnson (1994, eq. 11) at r for error at most eps,

        p_duo(r) = sum_{j,g} Pois(j; lam_1/2) Pois(g; lam_0/2) I_{1/2}(d/2 + g, d/2 + j),

    each Poisson sum cut to its shortest window with at most eps/4 omitted per tail;
    since 0 <= I_{1/2} <= 1 the omitted terms total at most eps.
    Returns (number of terms, j window, g window, error against quadrature)."""
    from scipy.special import betainc
    from scipy.stats import poisson

    def window(mu, tol):
        k = np.arange(int(mu + 40 * np.sqrt(mu + 1) + 100))
        lo = int(np.flatnonzero(np.r_[0.0, poisson.cdf(k[:-1], mu)] <= tol).max())
        hi = int(np.flatnonzero(poisson.sf(k, mu) <= tol).min())
        return np.arange(lo, hi + 1)

    lam_1, lam_0 = lambdas(r)
    j, g = window(lam_1 / 2, eps / 4), window(lam_0 / 2, eps / 4)
    B = betainc(d / 2 + g[None, :], d / 2 + j[:, None], 0.5)
    p = poisson.pmf(j, lam_1 / 2) @ B @ poisson.pmf(g, lam_0 / 2)
    return len(j) * len(g), (j[0], j[-1]), (g[0], g[-1]), abs(p - pduo_quad(r, d, QUAD_FLOOR))


def simulate(r, n, d, rng):
    """Monte Carlo p_duo(r): the reference x0, its repeat x1 and the odd stimulus y,
    all with identity covariance; correct when x1 is the nearer to x0."""
    delta = np.zeros(d)
    delta[0] = np.sqrt(r)
    x0 = rng.standard_normal((n, d))
    x1 = rng.standard_normal((n, d))
    y = delta + rng.standard_normal((n, d))
    return np.mean(np.sum((x1 - x0) ** 2, 1) < np.sum((y - x0) ** 2, 1))
