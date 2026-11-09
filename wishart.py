"""The covariance-field model of hong2026 and the two fits of Figure 3, d = 2.

    Sigma(x) = U(x) U(x)^T + diag_term I,    U(x) = sum_{a,b} W_{ab} T_a(x_1) T_b(x_2),

a Chebyshev expansion of degree 5 in each stimulus coordinate.  W has shape
(5, 5, d, d + 1) and each entry has the prior N(0, variance_scale decay^(a+b)).
Settings are those of the production fits: variance_scale 3e-4, decay 0.4,
diag_term 0, one extra column.

Two log posteriors over W for triangle-test trials (y, xref, xprobe), where y = 1
is a correct response, the two reference presentations have covariance
Sigma(xref) and the probe has Sigma(xprobe):

  nn_log_posterior          p_tri by the network of triangle.py; exact Bernoulli terms
                            from its log odds
  mc_objective              p_tri by simulating the trial with 2000 samples, the
                            indicator smoothed by a logistic kernel of bandwidth
                            5e-3 so that it is differentiable in W, and clipped to
                            [0.001, 0.999]: the objective of hong2026's code, which
                            is -(mean log-likelihood + log prior / number of trials)

and the two fits: L-BFGS-B on the first, SGD with momentum on the second.
"""
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import triangle as T

DEGREE, EXTRA, VARIANCE_SCALE, DECAY, DIAG = 5, 1, 3e-4, 0.4, 0.0
D = 2


# ----------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------
def chebyshev_T(x):
    """T_0(x), ..., T_{DEGREE-1}(x) by the recurrence; x of shape (n,) -> (n, DEGREE)."""
    Ts = [jnp.ones_like(x), x]
    for _ in range(2, DEGREE):
        Ts.append(2.0 * x * Ts[-1] - Ts[-2])
    return jnp.stack(Ts[:DEGREE], axis=-1)


def prior_sd():
    """sd of W_{ab..}: sqrt(variance_scale decay^(a+b)), shape (5, 5, 1, 1)."""
    a = jnp.arange(DEGREE)
    return jnp.sqrt(VARIANCE_SCALE * DECAY ** (a[:, None] + a[None, :]))[..., None, None]


def sample_W(key):
    return prior_sd() * jax.random.normal(key, (DEGREE, DEGREE, D, D + EXTRA))


def log_prior(W):
    return jax.scipy.stats.norm.logpdf(W, scale=prior_sd()).sum()


def U_field(W, x):
    """U(x) for x of shape (n, 2): (n, D, D + EXTRA).  x is clipped to [-1, 1]^2."""
    xt = jnp.clip(x, -1.0, 1.0)
    phi = chebyshev_T(xt[:, 0])[:, :, None] * chebyshev_T(xt[:, 1])[:, None, :]   # (n, 5, 5)
    return jnp.einsum('abdv,iab->idv', W, phi)


def Sigma_field(W, x):
    U = U_field(W, x)
    return jnp.einsum('ijk,ihk->ijh', U, U) + DIAG * jnp.eye(D)


def grid(cells=40, half=0.7):
    """Cell midpoints of a uniform partition of [-half, half]^2, (cells^2, 2)."""
    e = jnp.linspace(-half, half, cells + 1)
    c = 0.5 * (e[:-1] + e[1:])
    return jnp.stack(jnp.meshgrid(c, c, indexing='ij'), axis=-1).reshape(-1, 2)


def field_distance(W, W_ref, x):
    """The relative error in Sigma of the paper: root mean square over x of
    ||Sigma(x; W) - Sigma(x; W_ref)||_F / ||Sigma(x; W_ref)||_F."""
    A, B = Sigma_field(W, x), Sigma_field(W_ref, x)
    rel = jnp.linalg.norm(A - B, axis=(1, 2)) / jnp.linalg.norm(B, axis=(1, 2))
    return float(jnp.sqrt(jnp.mean(rel ** 2)))


# ----------------------------------------------------------------------
# the log posterior with p_tri from the network, and its L-BFGS-B fit
# ----------------------------------------------------------------------
def nn_log_posterior(trials, net):
    y, xref, xprobe = (jnp.asarray(v) for v in trials)

    @jax.jit
    def log_posterior(W):
        z = T.logit_p_tri(Sigma_field(W, xref), Sigma_field(W, xprobe), xref, xprobe, net)
        log_lik = jnp.sum(y * jax.nn.log_sigmoid(z) + (1.0 - y) * jax.nn.log_sigmoid(-z))
        return log_lik + log_prior(W)
    return log_posterior


def fit_lbfgs(log_posterior, W0, maxiter=2000):
    """scipy L-BFGS-B on -log_posterior from W0, at scipy's default tolerances.

    Run twice: once with a callback recording every iterate and its wall time, after
    paying the XLA compile; once clean, whose wall time is `total`.
    """
    from scipy.optimize import minimize
    vg = jax.jit(jax.value_and_grad(lambda W: -log_posterior(W)))
    shape = np.shape(W0)
    last = {}

    def f(w):
        v, g = vg(jnp.asarray(w).reshape(shape))
        last['v'] = float(v)
        return last['v'], np.asarray(g, float).ravel()

    w0 = np.asarray(W0, float).ravel()
    t0 = time.time()
    f(w0)
    t_compile = time.time() - t0
    obj, wall, snaps = [], [], []
    t1 = time.time()

    def cb(w):
        snaps.append(np.asarray(w, float).reshape(shape))
        obj.append(last['v'])
        wall.append(time.time() - t1)

    minimize(f, w0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter}, callback=cb)
    t2 = time.time()
    r = minimize(f, w0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter})
    return dict(W=r.x.reshape(shape), total=time.time() - t2, n_iter=int(r.nit),
                t_compile=t_compile, obj=np.array(obj), wall=np.array(wall), snaps=np.array(snaps))


# ----------------------------------------------------------------------
# the Monte Carlo objective of hong2026 and its SGD fit
# ----------------------------------------------------------------------
def simulate_trial(Uref, Uprobe, mref, mprobe, key, n_samples):
    """n_samples simulated presentations of one trial: z0, z1 from the reference, z2
    from the probe.  Returns rho(z0, z1) - min(rho(z0, z2), rho(z1, z2)) in the metric
    of S = (2/3) Sigma_ref + (1/3) Sigma_probe: negative when the odd one is identified."""
    d, dv = Uref.shape
    k = jax.random.split(key, 6)
    z0 = jax.random.normal(k[0], (n_samples, dv)) @ Uref.T + mref + jnp.sqrt(DIAG) * jax.random.normal(k[3], (n_samples, d))
    z1 = jax.random.normal(k[1], (n_samples, dv)) @ Uref.T + mref + jnp.sqrt(DIAG) * jax.random.normal(k[4], (n_samples, d))
    z2 = jax.random.normal(k[2], (n_samples, dv)) @ Uprobe.T + mprobe + jnp.sqrt(DIAG) * jax.random.normal(k[5], (n_samples, d))
    S = (2.0 / 3.0) * (Uref @ Uref.T + DIAG * jnp.eye(d)) + (1.0 / 3.0) * (Uprobe @ Uprobe.T + DIAG * jnp.eye(d))
    r01, r02, r12 = z0 - z1, z0 - z2, z1 - z2
    X01, X02, X12 = jnp.split(jnp.linalg.solve(S, jnp.concatenate([r01.T, r02.T, r12.T], axis=1)), 3, axis=1)
    q01 = jnp.sum(r01 * X01.T, axis=1)
    q02 = jnp.sum(r02 * X02.T, axis=1)
    q12 = jnp.sum(r12 * X12.T, axis=1)
    return q01 - jnp.minimum(q02, q12)


simulate_trials = jax.vmap(simulate_trial, (0, 0, 0, 0, 0, None))


def mc_objective(trials, n_samples=2000, bandwidth=5e-3):
    """(value, gradient) of the Monte Carlo objective at W with random key."""
    y, xref, xprobe = (jnp.asarray(v) for v in trials)
    n = len(y)

    def objective(W, key):
        out = simulate_trials(U_field(W, xref), U_field(W, xprobe), xref, xprobe,
                              jax.random.split(key, n), n_samples)            # (n, n_samples)
        p = jnp.mean(jax.lax.logistic(-out / bandwidth), axis=1)              # smoothed P(out < 0)
        p = jnp.clip(p, 0.001, 0.999)
        return -(jnp.mean(y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p)) + log_prior(W) / n)
    return jax.jit(jax.value_and_grad(objective))


def fit_sgd(value_and_grad, W0, n_steps=1500, lr=1e-4, momentum=0.2, seed=1, snap_every=10):
    """SGD with momentum at the production settings, with the explosion guard of
    hong2026's optimizer: on NaN or a hundredfold blow-up, return to the best W seen
    and halve the learning rate.  Records the objective and wall time of every step
    and a W snapshot every snap_every steps."""
    import optax
    W = jnp.asarray(W0)
    key = jax.random.PRNGKey(seed)
    opt = optax.sgd(learning_rate=lr, momentum=momentum)
    state = opt.init(W)
    safe_W, min_val = W, np.inf
    obj, wall, snaps, snap_step = np.zeros(n_steps), np.zeros(n_steps), [], []
    t0 = time.time()
    for t in range(n_steps):
        key, _ = jax.random.split(key)
        val, grad = value_and_grad(W, key)
        val = float(val)
        if np.isnan(val) or (min_val > 0 and val > 1e2 * min_val) \
                or (min_val < 0 and val > min_val + 1e2 * abs(min_val)):
            W, lr = safe_W, 0.5 * lr
            opt = optax.sgd(learning_rate=lr, momentum=momentum)
            state = opt.init(W)
        else:
            updates, state = opt.update(grad, state)
            W = optax.apply_updates(W, updates)
            if val < min_val:
                safe_W, min_val = W, val
        obj[t], wall[t] = val, time.time() - t0
        if t % snap_every == 0:
            snaps.append(np.asarray(W))
            snap_step.append(t)
    return dict(W=np.asarray(W), obj=obj, wall=wall, snaps=np.array(snaps), snap_step=np.array(snap_step))


def mc_score(trials, Ws, n_batch=5000, n_rep=20, seed=7000):
    """The log posterior at each W in Ws by Monte Carlo p_tri with the hard rule, no
    smoothing and no clip, n_batch * n_rep samples per trial, the same random numbers
    at every W.  Returns the log posteriors (K,) and the standard error of
    logpost[0] - logpost[a] by the delta method from the per-trial and joint counts."""
    y, xref, xprobe = trials
    yj, xr, xp = jnp.asarray(y), jnp.asarray(xref), jnp.asarray(xprobe)
    n, K = len(y), len(Ws)
    U = [(U_field(jnp.asarray(W), xr), U_field(jnp.asarray(W), xp)) for W in Ws]

    @jax.jit
    def correct(Ur, Up, keys):
        return simulate_trials(Ur, Up, xr, xp, keys, n_batch) < 0

    c, cc = np.zeros((K, n)), np.zeros((K, K, n))
    for rep in range(n_rep):
        keys = jax.random.split(jax.random.PRNGKey(seed + rep), n)
        hits = [np.asarray(correct(Ur, Up, keys)) for Ur, Up in U]
        for a in range(K):
            c[a] += hits[a].sum(1)
            for b in range(K):
                cc[a, b] += (hits[a] & hits[b]).sum(1)
    N = n_batch * n_rep
    p, pj = c / N, cc / N
    q = np.where(y == 1, p, 1 - p)                       # probability of the observed response
    logpost = np.log(q).sum(1) + np.array([float(log_prior(jnp.asarray(W))) for W in Ws])
    g = np.where(y == 1, 1 / np.where(y == 1, p, 1.0), -1 / np.where(y == 1, 1.0, 1 - p))
    se = np.zeros(K)
    for a in range(1, K):
        var = (g[a] ** 2 * p[a] * (1 - p[a]) + g[0] ** 2 * p[0] * (1 - p[0])
               - 2 * g[a] * g[0] * (pj[a, 0] - p[a] * p[0])) / N
        se[a] = np.sqrt(var.sum())
    return logpost, se
