"""Evaluating p_tri and p_duo with the algorithms of the paper.

p_tri, the triangle test (Section 3): the exact reduction of the 2d + d(d+1)
arguments to 2d, then the network, for d = 2 or 3.
p_duo, the duo-trio test with Sigma_x = Sigma_y = Sigma (Section 4): the
Chebyshev expansion in r = (mu_y - mu_x)^T Sigma^{-1} (mu_y - mu_x), stored for d = 2 and 3.

Each is printed beside a reference value and the difference: Monte Carlo for
p_tri, adaptive quadrature for p_duo.

    python example.py
"""
import numpy as np
import jax
import jax.numpy as jnp

import triangle as T
import duotrio as D

rng = np.random.default_rng(0)
n = 5                                       # trials per batch
n_mc = 10**6                                # Monte Carlo samples per trial


def random_spd(n, d):
    A = rng.normal(size=(n, d, d))
    return A @ np.swapaxes(A, 1, 2) + 0.1 * np.eye(d)        # (n, d, d)


# ----------------------------------------------------------------------
# p_tri: two presentations of N(mu_x, Sigma_x), one of N(mu_y, Sigma_y)
# ----------------------------------------------------------------------
for d in (2, 3):
    net = T.load_network(d)                 # weights/nn_{d}d_16x2.npz
    Sigma_x = jnp.asarray(random_spd(n, d))
    Sigma_y = jnp.asarray(random_spd(n, d))
    mu_x = jnp.zeros((n, d))
    mu_y = jnp.asarray(2.0 * rng.normal(size=(n, d)))

    p = np.asarray(T.p_tri(Sigma_x, Sigma_y, mu_x, mu_y, net))   # (n,)
    # For a likelihood use the log odds z = T.logit_p_tri(...):
    # log p = log_sigmoid(z) and log(1 - p) = log_sigmoid(-z) stay accurate near p = 1.
    # Both are differentiable with jax.grad and can be compiled with jax.jit.

    p_mc = T.p_tri_mc(Sigma_x, Sigma_y, mu_x, mu_y, jax.random.PRNGKey(0), n_mc)
    print(f'p_tri, d = {d}:  network, Monte Carlo at 1e6 samples (sd <= 5e-4), difference')
    for a, b in zip(p, p_mc):
        print(f'  {a:.4f}  {b:.4f}  {a - b:+.1e}')

# ----------------------------------------------------------------------
# p_duo: reference N(mu_x, Sigma) twice and the odd stimulus N(mu_y, Sigma)
# ----------------------------------------------------------------------
d = 3
Sigma = random_spd(n, d)
delta = 2.0 * rng.normal(size=(n, d))                        # mu_y - mu_x
r = np.einsum('ni,nij,nj->n', delta, np.linalg.inv(Sigma), delta)

p = D.pduo(r, d)          # the paper's stored expansion; D.expansion(d) builds one for any d
print(f'p_duo, d = {d}:  r, Chebyshev expansion, quadrature, difference')
for ri, a in zip(r, p):
    # quadrature at the tightest tolerance it meets, well below the expansion's error
    b = D.pduo_quad(ri, d, D.QUAD_FLOOR)
    print(f'  {ri:8.3f}  {a:.10f}  {b:.10f}  {a - b:+.1e}')
