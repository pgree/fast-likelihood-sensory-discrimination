# Fast likelihood evaluation for multivariate sensory discrimination models

Code and data that implement the algorithms in and reproduce the figures in

> P. Greengard, D. H. Brainard, F. Hong, A. Williams. Fast likelihood
> evaluation for multivariate sensory discrimination models. arXiv:XXXX.XXXXX, 2026.

To use the algorithms, start from `example.py`, a script that evaluates `p_tri`
with the neural network and `p_duo` with the Chebyshev expansion for several
randomly chosen arguments. The script also prints differences with Monte Carlo
or quadrature evaluation. The two calls are

```python
import triangle as T, duotrio as D

net = T.load_network(2)                          # d = 2 or 3
p = T.p_tri(Sigma_x, Sigma_y, mu_x, mu_y, net)   # (n, d, d), (n, d, d), (n, d), (n, d) -> (n,)

p = D.pduo(r, 3)                                 # d = 2 or 3; r = (mu_y - mu_x)^T Sigma^{-1} (mu_y - mu_x), (n,) -> (n,)
```

The code uses numpy, scipy, jax, optax and
matplotlib (the paper's numbers were produced with Python 3.13, jax 0.10.0,
numpy 2.4, scipy 1.17, optax 0.2.6).

| file | description |
|---|---|
| `example.py` | evaluating `p_tri` and `p_duo` for a few random trials, beside Monte Carlo and quadrature |
| `triangle.py` | `p_tri` for the triangle test: the exact reduction to 2d arguments (Section 3.1) and the 2d-16-16-1 network for the reduced function (Section 3.2); a Monte Carlo simulator of the trial for the checks |
| `duotrio.py` | `p_duo` for the duo-trio test with equal covariances: the one-dimensional integral (Claim 3), its Chebyshev expansion (Section 4.1), and the checks the paper quotes |
| `wishart.py` | the covariance-field model of hong2026 and the two log posteriors of the MAP comparison, with `p_tri` from the network and by Monte Carlo (Figure 4) |
| `reproduce.py` | prints every number in the paper beside the value computed here, and draws the three figures into `figures/` |
| `weights/` | the two networks the paper reports: 369 parameters for d = 2, 401 for d = 3 |
| `data/` | the frozen test and selection sets (4992 points each, labels at n = 1e7), the training sets (300000 points each at n = 1e5), and the cached fits of the MAP comparison |
| `supertests/` | supertests of the d = 2 network beyond the training region; `network.py` is the network in plain Python, written by `make_network.py` |

The MAP comparison in `reproduce.py` uses the trials of hong2026, which are not
included in this repository. To run it, pass them with `--trials PATH`, an npz with
`y` (n,), the responses in {0, 1}; `xref` (n, 2), the stimulus shown twice; and
`xprobe` (n, 2), the stimulus shown once. The cached runs use all n = 6000 trials of
the d = 2 experiment. Without `--trials` the section is skipped.

`supertests/tail.py` is an example of supertests for Schematic's Super, which
either proves that a property holds for every input or returns a counterexample.
Together its two supertests show that the d = 2 network stays within 2.2e-5 of
1 for every `||eta|| >= 12`. They need
`pip install schematic-supertest` and a Schematic account:

```bash
super login
super link
super check supertests/tail.py
```