"""p_tri for the triangle test (Section 3): the reduction g and the network h.

    p_tri(mu_x, mu_y, Sigma_x, Sigma_y) = h(g(mu_x, mu_y, Sigma_x, Sigma_y))

g   the exact reduction of Section 3.1 from 2d + d(d+1) arguments to 2d, by the
    affine invariance of Claim 1: translate so that mu_x = 0, whiten
    S = (2/3) Sigma_x + (1/3) Sigma_y to the identity with M = L^{-1} (S = L L^T),
    then diagonalize M Sigma_x M^T = V diag(lambda) V^T.  The reduced arguments
    are lambda (descending, in (0, 3/2)) and eta = |V^T M (mu_y - mu_x)|.
h   a 2d-16-16-1 multilayer perceptron, GeLU hidden units and a logistic output,
    on (log lambda, eta) standardized to zero mean and unit variance (Section 3.2).
    weights/ holds the networks the paper reports: 369 parameters for d = 2,
    401 for d = 3.

Shapes: Sigma (n, d, d), mu (n, d), reduced inputs (n, 2d).
"""
import os
from functools import partial

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, 'weights', 'nn_{d}d_16x2.npz')


# ----------------------------------------------------------------------
# g: the reduction to (lambda, eta)
# ----------------------------------------------------------------------
def reduce_inputs(Sigma_x, Sigma_y, mu_x, mu_y):
    """(lambda, eta) for a batch of trials."""
    S = (2.0 / 3.0) * Sigma_x + (1.0 / 3.0) * Sigma_y
    M = jnp.linalg.inv(jnp.linalg.cholesky(S))              # M S M^T = I
    A = M @ Sigma_x @ jnp.swapaxes(M, -1, -2)                # whitened Sigma_x
    lam, V = jnp.linalg.eigh(A)                              # ascending
    lam, V = lam[:, ::-1], V[:, :, ::-1]                     # descending, Remark 2
    delta = jnp.einsum('nij,nj->ni', M, mu_y - mu_x)
    eta = jnp.abs(jnp.einsum('nji,nj->ni', V, delta))       # |V^T M (mu_y - mu_x)|
    return jnp.concatenate([lam, eta], axis=-1)


def reduce_inputs_2d_closed(Sigma_x, Sigma_y, mu_x, mu_y):
    """reduce_inputs for d = 2 with the Cholesky factor and the eigenvectors
    written out.  The general routines spend most of their time on overhead at
    this size; the paper's 0.041 us per evaluation is with this form.  Agrees
    with reduce_inputs to about 1e-5 in the inputs (the eigenvector formula
    loses digits when the two eigenvalues nearly coincide)."""
    a = (2.0 / 3.0) * Sigma_x[:, 0, 0] + (1.0 / 3.0) * Sigma_y[:, 0, 0]
    b = (2.0 / 3.0) * Sigma_x[:, 0, 1] + (1.0 / 3.0) * Sigma_y[:, 0, 1]
    c = (2.0 / 3.0) * Sigma_x[:, 1, 1] + (1.0 / 3.0) * Sigma_y[:, 1, 1]

    # S = L L^T, M = L^{-1}, lower triangular
    l11 = jnp.sqrt(a)
    l21 = b / l11
    l22 = jnp.sqrt(c - l21 * l21)
    m11, m21, m22 = 1.0 / l11, -l21 / (l11 * l22), 1.0 / l22

    # A = M Sigma_x M^T
    s11, s12, s22 = Sigma_x[:, 0, 0], Sigma_x[:, 0, 1], Sigma_x[:, 1, 1]
    r11, r12 = m11 * s11, m11 * s12
    r21, r22 = m21 * s11 + m22 * s12, m21 * s12 + m22 * s22
    A11 = r11 * m11
    A12 = r11 * m21 + r12 * m22
    A22 = r21 * m21 + r22 * m22

    # eigenvalues of [[A11, A12], [A12, A22]], descending
    tr = A11 + A22
    disc = jnp.sqrt((A11 - A22) ** 2 + 4.0 * A12 * A12)
    lam1, lam2 = 0.5 * (tr + disc), 0.5 * (tr - disc)

    # unit eigenvector for lam1; any basis works when A is a multiple of the identity
    vx, vy = lam1 - A22, A12
    nrm = jnp.sqrt(vx * vx + vy * vy)
    degenerate = nrm < 1e-12 * jnp.maximum(jnp.abs(lam1), 1.0)
    safe = jnp.where(degenerate, 1.0, nrm)
    v1x = jnp.where(degenerate, 0.0, vx / safe)
    v1y = jnp.where(degenerate, 1.0, vy / safe)

    d0 = mu_y[:, 0] - mu_x[:, 0]
    d1 = mu_y[:, 1] - mu_x[:, 1]
    md0, md1 = m11 * d0, m21 * d0 + m22 * d1               # M (mu_y - mu_x)
    e1 = v1x * md0 + v1y * md1
    e2 = -v1y * md0 + v1x * md1
    return jnp.stack([lam1, lam2, jnp.abs(e1), jnp.abs(e2)], axis=-1)


def canonical(X):
    """The trial whose reduction is X = (lambda, eta): Sigma_x = diag(lambda),
    Sigma_y = 3I - 2 diag(lambda), mu_x = 0, mu_y = eta (proof of Claim 2)."""
    n, d = X.shape[0], X.shape[1] // 2
    lam, eta = X[:, :d], X[:, d:]
    I = np.eye(d)
    return (jnp.asarray(lam[:, :, None] * I), jnp.asarray((3.0 - 2.0 * lam)[:, :, None] * I),
            jnp.zeros((n, d)), jnp.asarray(eta))


# ----------------------------------------------------------------------
# h: the network
# ----------------------------------------------------------------------
def init_params(key, arch):
    """Glorot-initialized weights for layer widths arch, biases zero."""
    params = []
    for i in range(len(arch) - 1):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (arch[i], arch[i + 1])) * jnp.sqrt(2.0 / (arch[i] + arch[i + 1]))
        params.append((W, jnp.zeros(arch[i + 1])))
    return params


def mlp_logit(params, xs):
    """GeLU hidden layers; the input to the logistic output unit.  xs standardized."""
    z = xs
    for W, b in params[:-1]:
        z = jax.nn.gelu(z @ W + b)
    W, b = params[-1]
    return (z @ W + b)[..., 0]


def transform(x, name):
    """The coordinate change before standardizing: (log lambda, eta) or none."""
    if name == 'loglambda':
        d = x.shape[-1] // 2
        return jnp.concatenate([jnp.log(x[..., :d]), x[..., d:]], axis=-1)
    return x


def input_stats(X, name='loglambda'):
    """Mean and sd of each transformed input over the training points."""
    Xt = transform(jnp.asarray(X), name)
    return dict(mean=Xt.mean(0), std=Xt.std(0), transform=name)


def standardize(x, stats):
    return (transform(x, stats['transform']) - stats['mean']) / stats['std']


def logit_h(x, net):
    """log odds of h at reduced inputs x (n, 2d); net = (params, stats)."""
    params, stats = net
    return mlp_logit(params, standardize(x, stats))


def h(x, net):
    return jax.nn.sigmoid(logit_h(x, net))


def p_tri(Sigma_x, Sigma_y, mu_x, mu_y, net, reduce=reduce_inputs):
    return h(reduce(Sigma_x, Sigma_y, mu_x, mu_y), net)


def logit_p_tri(Sigma_x, Sigma_y, mu_x, mu_y, net, reduce=reduce_inputs):
    """log p - log(1 - p): log p = -softplus(-z) and log(1 - p) = -softplus(z)
    stay accurate where p rounds to 1, which a likelihood needs."""
    return logit_h(reduce(Sigma_x, Sigma_y, mu_x, mu_y), net)


def load_network(d, path=None):
    z = np.load(path or WEIGHTS.format(d=d))
    params = [(jnp.array(z[f'W_{i}']), jnp.array(z[f'b_{i}'])) for i in range(int(z['n_layers']))]
    stats = dict(mean=jnp.array(z['input_mean']), std=jnp.array(z['input_std']),
                 transform=str(z['transform']))
    return params, stats


def save_network(path, net):
    params, stats = net
    out = {f'W_{i}': np.asarray(W) for i, (W, _) in enumerate(params)}
    out.update({f'b_{i}': np.asarray(b) for i, (_, b) in enumerate(params)})
    np.savez(path, n_layers=len(params), input_mean=np.asarray(stats['mean']),
             input_std=np.asarray(stats['std']), transform=stats['transform'], **out)


def n_params(net):
    return sum(int(W.size) + int(b.size) for W, b in net[0])


# ----------------------------------------------------------------------
# Monte Carlo: simulate the trial and count
# ----------------------------------------------------------------------
def p_tri_mc(Sigma_x, Sigma_y, mu_x, mu_y, key, n_samples, chunk=200_000):
    """Monte Carlo p_tri at each of a batch of K trials, n_samples per trial.

    Two percepts from N(mu_x, Sigma_x), one from N(mu_y, Sigma_y); correct when
    the odd one is farthest from the other two in the metric of S^{-1}.  Samples
    are drawn in float32 and the hit counts kept in float64.
    """
    K, d = Sigma_x.shape[0], Sigma_x.shape[-1]
    S_inv = jnp.linalg.inv((2.0 / 3.0) * Sigma_x + (1.0 / 3.0) * Sigma_y).astype(jnp.float32)
    Lx = jnp.linalg.cholesky(Sigma_x).astype(jnp.float32)
    Ly = jnp.linalg.cholesky(Sigma_y).astype(jnp.float32)
    mx, my = mu_x.astype(jnp.float32), mu_y.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(1,))
    def hits(key, m):
        k1, k2, k3 = jax.random.split(key, 3)

        def draw(k, L, mu):
            z = jax.random.normal(k, (K, m, d), dtype=jnp.float32)
            return jnp.einsum('kij,kmj->kmi', L, z) + mu[:, None, :]

        x1, x2, y = draw(k1, Lx, mx), draw(k2, Lx, mx), draw(k3, Ly, my)

        def rho(a, b):
            r = a - b
            return jnp.einsum('kmi,kij,kmj->km', r, S_inv, r)

        ok = rho(x1, x2) < jnp.minimum(rho(x1, y), rho(x2, y))
        return ok.sum(axis=1).astype(jnp.float64)

    total, done = jnp.zeros(K, dtype=jnp.float64), 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        key, sub = jax.random.split(key)
        total = total + hits(sub, m)
        done += m
    return np.asarray(total / n_samples)
