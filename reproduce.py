"""Reproduce the numbers and figures of "Fast likelihood evaluation for sensory
discrimination models" from the contents of this directory.

Each line prints the value the paper quotes beside the value computed here.
The default run takes about five minutes:

  accuracy   the networks against the frozen test sets (Section 3.2)
  timing     evaluation time of the reduced network, of a network on the
             original arguments, and of Monte Carlo; Figure 2
  duotrio    the Chebyshev expansion: truncation radius, node count, error,
             the Ennis-Johnson term count, timings; Figure 4 (Section 4)
  map        the MAP comparison on the trials of hong2026 (Figure 3): the
             fast-likelihood arm is refit here, the Monte Carlo arm is read
             from the cached 44 minute run unless --map-mc.  The trials are
             experimental data and are not included; pass them with
             --trials PATH (an npz with y, xref, xprobe, see the README) or
             the section is skipped

Flags for the long runs, off by default:

  --tail          remeasure 1 - h on the ray that sets R (Section 3.2); about a minute
  --train         retrain the two networks, 15 min each; writes weights/retrained_*
  --reduction     retrain the networks on the original arguments of Figure 2, 1-2 h
  --map-mc        rerun the Monte Carlo arm of the MAP comparison, 44 min
  --map-score     rescore the MAP end points by Monte Carlo at 1e5 samples per trial
  --map-restarts  redo the search for theta*: 200 L-BFGS-B fits from prior draws, 3 min

Timings are wall clock on one CPU and depend on the machine; the paper's are
from an Apple M3 Max with JAX 0.10.0.

Run from anywhere:  python reproduce.py [--only SECTION ...] [flags]
"""
import argparse, json, os, sys, time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import triangle as T
import duotrio as D
import wishart as M

DATA = os.path.join(HERE, 'data')
WEIGHTS = os.path.join(HERE, 'weights')
FIGURES = os.path.join(HERE, 'figures')
K = 20_000                  # trials per timed call, large enough to be compute-bound


def line(label, paper, here, note=''):
    print(f'  {label:<52s} paper {paper:>12s}   here {here}{"   " + note if note else ""}')


def timeit(fn, *args, repeats=10):
    """Median wall time of fn(*args) after one warm-up call."""
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def figure(name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, which='major', lw=0.3, alpha=0.4)

    def save():
        os.makedirs(FIGURES, exist_ok=True)
        path = os.path.join(FIGURES, name)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  wrote', path)
    return ax, save


# ----------------------------------------------------------------------
# accuracy: the networks against the frozen test sets
# ----------------------------------------------------------------------
QUOTED = {2: dict(params='369', rms='7.9e-4', se='1.7e-5', mx='6.6e-3', sel='7.7e-4', lab='8.3e-4'),
          3: dict(params='401', rms='9.3e-4', se='1.8e-5', mx='7.0e-3', sel='9.0e-4', lab='8.3e-4')}


def errors(net, name):
    """Network minus reference label on a frozen set, and the labels."""
    z = np.load(os.path.join(DATA, f'{name}.npz'))
    return np.asarray(T.h(jnp.asarray(z['X']), net)) - z['y'], z


def accuracy(d):
    net = T.load_network(d)
    e, z = errors(net, f'bench_test_{d}d')
    spec = json.loads(str(z['spec']))
    y, N = z['y'], len(e)
    rng = np.random.default_rng(0)       # standard error of the rms: 2000 draws of N points with replacement
    draws = np.sqrt((e ** 2)[rng.integers(0, N, (2000, N))].mean(1))
    q = QUOTED[d]
    print(f'd = {d}: test set of {N} points, labels by Monte Carlo at n = {spec["n_mc"]:.0e}, '
          f'pilot at n = {spec["pilot_n_mc"]}')
    line('parameters', q['params'], f'{T.n_params(net)}')
    line('rms deviation from the reference values', q['rms'], f'{np.sqrt((e ** 2).mean()):.3e}')
    line('its standard error (2000 bootstrap draws)', q['se'], f'{draws.std():.2e}')
    line('maximum deviation', q['mx'], f'{np.abs(e).max():.3e}')
    line('sd of a reference value, bound and rms', '<= 1.6e-4',
         f'{np.sqrt(0.25 / spec["n_mc"]):.2e}, {np.sqrt((y * (1 - y)).mean() / spec["n_mc"]):.2e}')
    es, _ = errors(net, f'bench_selection_{d}d')
    line('rms on the selection set (Figure 2)', q['sel'], f'{np.sqrt((es ** 2).mean()):.3e}')
    zt = np.load(os.path.join(DATA, f'train_domain_radial_{d}d.npz'))
    yt, n_mc = zt['train_labels'], int(zt['n_mc'])
    line(f'training labels: {len(yt)} points at n = {n_mc:.0e}, rms sd', q['lab'],
         f'{np.sqrt((yt * (1 - yt)).mean() / n_mc):.2e}', f'(max {np.sqrt(0.25 / n_mc):.2e}; paper: bounded by 0.002)')


# ----------------------------------------------------------------------
# timing: the reduced network, a network on the original arguments, Monte Carlo
# ----------------------------------------------------------------------
N_MC = [300, 1000, 3000, 10_000, 30_000, 100_000]
LADDER = [(16, 2), (32, 2), (64, 2), (128, 2), (128, 3)]


def trials(rng, d):
    """K trials in the original variables whose reductions cover the domain."""
    lam = rng.uniform(1e-3, 1.499, (K, d))
    A = rng.normal(size=(K, d, d))
    Sx = np.einsum('kij,kj,klj->kil', A, lam, A)
    Sy = np.einsum('kij,kj,klj->kil', A, 3 - 2 * lam, A)
    u = np.abs(rng.normal(size=(K, d)))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    my = u * rng.uniform(0, 12, K)[:, None]
    return tuple(jnp.asarray(v) for v in (Sx, Sy, np.zeros((K, d)), my))


def raw_network(width, depth, d=2, net=None):
    """A network on the d(d+1) + d original arguments (random weights unless given)
    and the function assembling its inputs from a batch of trials."""
    n_in = d * (d + 1) + d
    if net is None:
        net = (T.init_params(jax.random.PRNGKey(0), (n_in,) + (width,) * depth + (1,)),
               dict(mean=jnp.zeros(n_in), std=jnp.ones(n_in), transform='identity'))
    iu = np.triu_indices(d)

    def inputs(Sx, Sy, mx, my):
        return jnp.concatenate([Sx[:, iu[0], iu[1]], Sy[:, iu[0], iu[1]], my - mx], axis=-1)
    return net, inputs


def ladder_errors():
    """(parameters, rms on the selection set) of the networks on the original arguments,
    recomputed by --reduction if that has been run, else from the paper's cache."""
    path = os.path.join(DATA, 'reduction_gain_2d_recomputed.npz')
    src = 'recomputed by --reduction' if os.path.exists(path) else "the paper's cache"
    z = np.load(path if os.path.exists(path) else os.path.join(DATA, 'reduction_gain_2d.npz'))
    return {(w, dp): (int(z[f'raw_{w}x{dp}'][0]), float(z[f'raw_{w}x{dp}'][1])) for w, dp in LADDER}, src


def timing():
    rng = np.random.default_rng(0)
    Sx, Sy, mx, my = trials(rng, 2)
    net = T.load_network(2)
    f_closed = jax.jit(lambda a, b, c, e: T.p_tri(a, b, c, e, net, reduce=T.reduce_inputs_2d_closed))
    f_eigh = jax.jit(lambda a, b, c, e: T.p_tri(a, b, c, e, net))
    t_closed = timeit(f_closed, Sx, Sy, mx, my) / K
    t_eigh = timeit(f_eigh, Sx, Sy, mx, my) / K
    print(f'd = 2, batches of {K} trials, median of 10 calls')
    line('reduced network, closed-form reduction', '0.041 us', f'{t_closed * 1e6:.3f} us')
    line('reduced network, reduction by cholesky and eigh', '-', f'{t_eigh * 1e6:.3f} us', '(not quoted)')
    raw_t = {}
    for width, depth in LADDER:
        rnet, rin = raw_network(width, depth)
        f_raw = jax.jit(lambda a, b, c, e: T.h(rin(a, b, c, e), rnet))
        raw_t[(width, depth)] = (timeit(f_raw, Sx, Sy, mx, my) / K, T.n_params(rnet))
    t_raw, m_raw = raw_t[(128, 3)]
    line('network on the original arguments, 128x3', '0.28 us', f'{t_raw * 1e6:.3f} us', f'({m_raw} parameters)')
    line('its time relative to the reduced network', 'about 7x', f'{t_raw / t_closed:.1f}x')
    line('its parameters relative to the reduced network', 'two orders', f'{m_raw / T.n_params(net):.0f}x')

    # Monte Carlo: throughput measured where sampling dominates dispatch and the cost
    # of an n-sample evaluation charged as n / throughput, a lower bound on its cost
    n_sat = 10_000
    t_sat = timeit(lambda a, b, c, e: T.p_tri_mc(a, b, c, e, jax.random.PRNGKey(0), n_sat, chunk=2000),
                   Sx, Sy, mx, my)
    per_sample = t_sat / (K * n_sat)
    line('Monte Carlo at n = 1e5, time per evaluation', '1.5 ms', f'{1e5 * per_sample * 1e3:.2f} ms',
         f'({1 / per_sample:.2e} samples per second)')
    zt = np.load(os.path.join(DATA, 'bench_test_2d.npz'))
    p = T.p_tri_mc(*T.canonical(zt['X']), jax.random.PRNGKey(1), 100_000, chunk=10_000)
    line('Monte Carlo at n = 1e5, rms error on the test set', '1.3e-3', f'{np.sqrt(np.mean((p - zt["y"]) ** 2)):.2e}')

    Sx3, Sy3, mx3, my3 = trials(rng, 3)
    net3 = T.load_network(3)
    f3 = jax.jit(lambda a, b, c, e: T.p_tri(a, b, c, e, net3))
    print(f'd = 3: reduced network with cholesky and eigh {timeit(f3, Sx3, Sy3, mx3, my3) / K * 1e6:.3f} us (not quoted)')

    # Figure 2: errors on the selection set against parameters and against time
    es, zs = errors(net, 'bench_selection_2d')
    rms_red = float(np.sqrt((es ** 2).mean()))
    ladder, src = ladder_errors()
    print(f'  Figure 2 uses the errors of the networks on the original arguments from {src}')
    mc_rms = []
    for n in N_MC:                       # one n-sample estimate at each selection point
        p = T.p_tri_mc(*T.canonical(zs['X']), jax.random.PRNGKey(2 + n), n, chunk=10_000)
        mc_rms.append(float(np.sqrt(np.mean((p - zs['y']) ** 2))))
    raw = np.array([[ladder[k][0], raw_t[k][0], ladder[k][1]] for k in LADDER])   # M, time, rms
    ax, save = figure('reduction_gain.pdf')
    ax.plot(raw[:, 0], raw[:, 2], '-', marker='o', ms=4, color='tab:blue')
    ax.plot(T.n_params(net), rms_red, marker='*', ms=13, ls='none', color='tab:red')
    ax.text(raw[0, 0] * 1.15, raw[0, 2] * 1.35, r'$d^2 + 2d$ arguments', color='tab:blue', fontsize=8)
    ax.text(T.n_params(net) * 1.25, rms_red, r'$2d$ arguments', color='tab:red', fontsize=8, va='center')
    ax.set_xlim(2.4e2, 1.6e5)
    ax.set_ylim(5e-4, 5e-2)
    ax.set_xlabel('parameters')
    ax.set_ylabel('RMS error')
    save()
    ax, save = figure('time_vs_error.pdf')
    mc_t = np.array(N_MC) * per_sample * 1e6
    ax.plot(mc_t, mc_rms, '-', marker='s', ms=3.5, color='tab:green')
    ax.plot(raw[:, 1] * 1e6, raw[:, 2], '-', marker='o', ms=4, color='tab:blue')
    ax.plot(t_closed * 1e6, rms_red, marker='*', ms=13, ls='none', color='tab:red')
    ax.text(mc_t[0] * 1.2, mc_rms[0] * 1.13, 'Monte Carlo', color='tab:green', fontsize=8)
    ax.text(raw[0, 1] * 1e6, raw[0, 2] * 1.18, r'$d^2 + 2d$ arguments', color='tab:blue', fontsize=8)
    ax.text(t_closed * 1e6 * 1.45, rms_red * 0.95, r'$2d$ arguments', color='tab:red', fontsize=8, va='top')
    ax.set_ylim(5e-4, 5e-2)
    ax.set_xlabel(r'time for one evaluation ($\mu$s)')
    ax.set_ylabel('RMS error')
    save()


# ----------------------------------------------------------------------
# duo-trio: the Chebyshev expansion
# ----------------------------------------------------------------------
N_MC_DUO = [30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000]


def duotrio():
    eps, n = 1e-8, 36
    tables = {}
    for d, R_paper in ((3, '220.75'), (2, '218.375')):
        b = D.build(d, eps, n)
        tables[d] = b
        print(f'd = {d}: expansion built at eps = {eps:g} with {n} nodes')
        line('truncation radius R', R_paper, f'{b["r_max"]:g}')
        line('1 - pduo(R)', '1.0e-9' if d == 3 else '9.9e-10', f'{D.one_minus_pduo_quad(b["r_max"], d):.2e}')
        line('node values against tighter quadrature', '< 1e-9', f'{b["quad_err"]:.1e}')
        line('coefficients dropped (tail of a 72-node build)', '< 1e-9', f'{b["tail"]:.1e}',
             f'(sampled between nodes {b["sampled"]:.1e})')
        line('stored coefficients (TABLE) against this build', '-',
             f'{np.abs(b["coef"] - D.TABLE[d][1]).max():.1e}', f'(r_max {D.TABLE[d][0]:g})')
    R, coef = tables[3]['r_max'], tables[3]['coef']
    diff, at = D.doubling(3, coef, R, eps)
    line('d = 3: sup error against the 72-node expansion', '8.7e-12', f'{diff:.2e}', f'(at r = {at:.3g})')
    for d in (3, 2):
        cnt, jw, gw, err = D.series_terms(d, tables[d]['r_max'], eps)
        line(f'd = {d}: Ennis-Johnson terms at r = R for error 1e-8', '2328' if d == 3 else '-', f'{cnt}',
             f'(j = {jw[0]}..{jw[1]}, g = {gw[0]}..{gw[1]}; error against quadrature {err:.1e})')

    rng = np.random.default_rng(0)
    r = rng.uniform(0.0, R, K)
    t = timeit(lambda v: D.evaluate(v, coef, R), r, repeats=20) / K
    line('expansion, time per evaluation', '0.028 us', f'{t * 1e6:.3f} us', f'(batch of {K})')
    t0 = time.perf_counter()
    for v in r[:200]:
        D.pduo_quad(v, 3, 1e-10)
    tq = (time.perf_counter() - t0) / 200
    line('adaptive quadrature at 1e-10, time per evaluation', '28 ms', f'{tq * 1e3:.1f} ms')
    r_ref = r[:400]
    ref = np.array([D.pduo_quad(v, 3, D.QUAD_FLOOR) for v in r_ref])
    err = float(np.sqrt(np.mean((D.evaluate(r_ref, coef, R) - ref) ** 2)))
    line('rms of the expansion against quadrature, 400 points', '-', f'{err:.1e}', '(Figure 4)')

    # Figure 4: Monte Carlo, both axes measured; 20 estimates at each reference point
    reps = 20
    mc = []
    for n_mc in N_MC_DUO:
        tt = timeit(lambda: D.simulate(4.0, n_mc, 3, rng), repeats=max(5, min(20, int(2e6 / n_mc) + 1)))
        sq = sum(float(((np.array([D.simulate(rj, n_mc, 3, rng) for _ in range(reps)]) - pj) ** 2).sum())
                 for rj, pj in zip(r_ref, ref))
        mc.append([tt, np.sqrt(sq / (len(r_ref) * reps))])
    mc = np.array(mc)
    ax, save = figure('duotrio_time_accuracy.pdf')
    ax.plot(mc[:, 0] * 1e6, mc[:, 1], '-', marker='s', ms=3.5, color='tab:green')
    ax.plot(t * 1e6, err, marker='*', ms=13, ls='none', color='tab:red')
    ax.text(mc[3, 0] * 1e6, mc[3, 1] / 2.6, 'Monte Carlo', color='tab:green', fontsize=8, ha='center', va='top')
    ax.text(t * 1e6 * 1.6, err, 'Chebyshev', color='tab:red', fontsize=8, va='center')
    ax.set_ylim(err / 5.0, None)
    ax.set_xlabel(r'time for one evaluation ($\mu$s)')
    ax.set_ylabel('RMS error')
    save()


# ----------------------------------------------------------------------
# the MAP comparison on the trials of hong2026
# ----------------------------------------------------------------------
def load_trials(path):
    """(y, xref, xprobe): responses in {0, 1} and the two stimuli of each trial, (n,), (n, 2), (n, 2)."""
    z = np.load(path)
    return z['y'].astype(float), z['xref'], z['xprobe']


def map_comparison(trials_, rerun_mc=False, rescore=False, restarts=False):
    net = T.load_network(2)
    z = np.load(os.path.join(DATA, 'map_timing_local_2d.npz'), allow_pickle=True)
    log_posterior = M.nn_log_posterior(trials_, net)
    x = M.grid()
    W_star, W0 = jnp.asarray(z['W_star']), jnp.asarray(z['W0'])
    L_star = float(log_posterior(W_star))

    def gap(W):          # log p(theta* | y) - log p(theta | y), Figure 3a
        return L_star - float(log_posterior(jnp.asarray(W)))

    def dist(W):         # relative error in Sigma(x) over the grid, Figure 3b
        return M.field_distance(jnp.asarray(W), W_star, x)

    y = trials_[0]
    assert len(y) == int(z['n_trials']), f'the cached run used {int(z["n_trials"])} trials, the file has {len(y)}'
    print(f'd = 2: {len(y)} of {int(z["n_trials_full"])} trials, both arms from theta* + {float(z["perturb"])} xi')
    line('log posterior at theta* (cached search over 200 starts)', '-', f'{L_star:.4f}',
         f'(cache: {-float(z["nn_0_obj"][-1]) + gap(z["nn_0_W"]):.4f})')
    line('start: nats below theta*, relative error in Sigma', '-', f'{gap(W0):.1f}, {dist(W0):.3f}')
    line('fast likelihood, L-BFGS-B (cached run): time', '0.30 s', f'{float(z["nn_0_total"]):.2f} s',
         f'({int(z["nn_0_n_iter"])} iterations)')
    line('    nats below theta*, relative error in Sigma', '0.053, 0.008', f'{gap(z["nn_0_W"]):.3f}, {dist(z["nn_0_W"]):.4f}')
    mc_run = dict(W=z['mc_0_W'], wall=z['mc_0_wall'], snaps=z['mc_0_snaps'], snap_step=z['mc_0_snap_step'])
    line('Monte Carlo likelihood, SGD (cached run): time', '2618 s', f'{float(mc_run["wall"][-1]):.0f} s',
         f'({int(z["mc_steps"])} steps, {int(z["mc_samples"])} samples per trial)')
    line('    nats below theta*, relative error in Sigma', '0.76, 0.029', f'{gap(mc_run["W"]):.4f}, {dist(mc_run["W"]):.4f}')

    r = M.fit_lbfgs(log_posterior, W0)
    line('fast likelihood refit now from the same start: time', '0.30 s', f'{r["total"]:.2f} s',
         f'({r["n_iter"]} iterations; compile {r["t_compile"]:.2f} s not included)')
    line('    nats below theta*, relative error in Sigma', '0.053, 0.008', f'{gap(r["W"]):.3f}, {dist(r["W"]):.4f}')

    if rerun_mc:
        print('  rerunning the Monte Carlo arm (44 min) ...', flush=True)
        mc_run = M.fit_sgd(M.mc_objective(trials_), W0)
        line('Monte Carlo likelihood rerun: time', '2618 s', f'{mc_run["wall"][-1]:.0f} s')
        line('    nats below theta*, relative error in Sigma', '0.76, 0.029', f'{gap(mc_run["W"]):.3f}, {dist(mc_run["W"]):.4f}')
        np.savez(os.path.join(DATA, 'map_timing_mc_rerun_2d.npz'), **mc_run)

    if rescore:
        print('  scoring theta* and both end points by Monte Carlo at 1e5 samples per trial ...', flush=True)
        lp, se = M.mc_score(trials_, [W_star, mc_run['W'], r['W']])
    else:
        zs = np.load(os.path.join(DATA, 'map_timing_mcscore_2d.npz'), allow_pickle=True)
        lp, se = zs['logpost_mc'], zs['gap_se']
        print('  end points scored by Monte Carlo at 1e5 samples per trial (cached):')
    line('    fast-likelihood end: nats below theta*', '-', f'{lp[0] - lp[2]:+.3f}', f'(standard error {se[2]:.3f})')
    line('    Monte Carlo end: nats below theta*', '-', f'{lp[0] - lp[1]:+.3f}', f'(standard error {se[1]:.3f})')

    if restarts:
        print('  searching for theta*: 200 L-BFGS-B fits from prior draws ...', flush=True)
        vals = []
        t0 = time.time()
        for i in range(200):
            vals.append(-float(log_posterior(jnp.asarray(M.fit_lbfgs(log_posterior, M.sample_W(jax.random.PRNGKey(10_000 + i)))['W']))))
        vals = np.array(vals)
        line('best of 200 restarts, log posterior', f'{L_star:.4f}', f'{-vals.min():.4f}',
         f'({(vals <= vals.min() + 0.01).sum()} of 200 within 0.01 nats; {time.time() - t0:.0f} s)')

    # Figure 3: both arms along their recorded runs; the fast arm is the refit above
    ax_g, save_g = figure('map_timing_gap.pdf')
    ax_s, save_s = figure('map_timing_sigma.pdf')
    for run, colour, label, wall in ((mc_run, 'tab:green', 'MC likelihood', mc_run['wall'][mc_run['snap_step']]),
                                     (r, 'tab:red', 'fast likelihood', r['wall'])):
        # end each curve at the final iterate, which the text quotes; the Monte
        # Carlo snapshots stop at step 1490 of 1500
        snaps = list(run['snaps']) + [run['W']]
        wall = np.append(wall, run['wall'][-1])
        gaps = np.array([gap(W) for W in snaps])
        dists = np.array([dist(W) for W in snaps])
        ok = gaps > 0
        ax_g.plot(wall[ok], gaps[ok], '-', color=colour, lw=1.5)
        ax_s.plot(wall, dists, '-', color=colour, lw=1.5)
        ax_g.text(wall[len(wall) // 2], gaps[len(wall) // 2] * 3, label, color=colour, fontsize=8)
        ax_s.text(wall[len(wall) // 2], dists[len(wall) // 2] * 3, label, color=colour, fontsize=8)
    ax_g.set_xlabel('time (s)')
    ax_g.set_ylabel(r'$\log p(\hat\theta\,|\,y) - \log p(\theta\,|\,y)$')
    ax_s.set_xlabel('time (s)')
    ax_s.set_ylabel(r'relative error in $\Sigma(x)$')
    save_g()
    save_s()


# ----------------------------------------------------------------------
# the ray that sets R (Section 3.2), remeasured
# ----------------------------------------------------------------------
RADII = np.array([4., 5., 6., 7., 8., 9., 10., 11., 12., 14.])


def mc(X, n, seed):
    return T.p_tri_mc(*T.canonical(X), jax.random.PRNGKey(seed), n, chunk=max(1000, int(2e7 // len(X))))


def tail(d):
    # the lambda maximizing 1 - h at ||eta|| = 9 over a 13-grid on the ordered wedge
    # and three directions, then 1 - h along its ray at 5e7 samples
    g = np.linspace(1e-3, 1.5 - 1e-3, 13)
    lam = np.array([p for p in np.stack(np.meshgrid(*[g] * d, indexing='ij'), -1).reshape(-1, d)
                    if np.all(np.diff(p) <= 0)])
    dirs = [np.eye(d)[0], np.eye(d)[-1], np.ones(d) / np.sqrt(d)]
    X = np.concatenate([np.column_stack([lam, np.tile(u, (len(lam), 1)) * 9.0]) for u in dirs])
    q = 1.0 - mc(X, 1_000_000, 101)
    i = int(np.argmax(q))
    lam_w, u_w = X[i, :d], X[i, d:] / np.linalg.norm(X[i, d:])
    t = 1.0 - mc(np.column_stack([np.tile(lam_w, (len(RADII), 1)), np.outer(RADII, u_w)]), 50_000_000, 7)
    print(f'd = {d}: worst lambda at ||eta|| = 9 is {np.round(lam_w, 3)}, direction {np.round(u_w, 2)} '
          f'(paper: near (3/2, ..., 3/2))')
    print('  ||eta||   ' + ' '.join(f'{r:8.0f}' for r in RADII))
    print('  1 - h     ' + ' '.join(f'{v:8.1e}' for v in t))
    k = int(np.flatnonzero(t < 1e-4)[0])      # R at 1e-4, interpolating linearly between grid radii
    R = RADII[k - 1] + (RADII[k] - RADII[k - 1]) * (t[k - 1] - 1e-4) / (t[k - 1] - t[k])
    line('R with 1 - h < 1e-4', '10.9' if d == 2 else '11.1', f'{R:.1f}')
    line('1 - h at ||eta|| = 12', '< 2.2e-5', f'{t[RADII == 12.0][0]:.2e}', f'(sd {np.sqrt(t[RADII == 12.0][0] / 5e7):.1e})')


# ----------------------------------------------------------------------
# training (Section 3.2): cross-entropy, Adam, cosine decay, best of restarts
# ----------------------------------------------------------------------
N_STEPS, N_VAL, BATCH, LR0 = 112_000, 10_000, 1024, 1e-3


def train_network(X, y, arch, key, name='loglambda'):
    """One training run on (X, y): the recipe of the paper.  Returns (params, stats)."""
    import optax
    stats = T.input_stats(X, name)
    Xs, yj = T.standardize(jnp.asarray(X), stats), jnp.asarray(y)
    key, k = jax.random.split(key)
    params = T.init_params(k, arch)
    opt = optax.adam(optax.cosine_decay_schedule(LR0, N_STEPS, alpha=1e-2))
    state = opt.init(params)

    def loss(params, x, y):
        p = jnp.clip(jax.nn.sigmoid(T.mlp_logit(params, x)), 1e-9, 1.0 - 1e-9)
        return -jnp.mean(y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p))

    @jax.jit
    def step(params, state, x, y):
        g = jax.grad(loss)(params, x, y)
        updates, state = opt.update(g, state)
        return optax.apply_updates(params, updates), state

    n = len(Xs)
    steps_per_epoch = n // BATCH
    for _ in range(round(N_STEPS / steps_per_epoch)):
        key, k = jax.random.split(key)
        perm = jax.random.permutation(k, n)
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            params, state = step(params, state, Xs[idx], yj[idx])
    return params, stats


def best_of(X, y, arch, restarts, name, label):
    """Best of `restarts` training runs by empirical risk on the training labels."""
    best = None
    for seed in range(restarts):
        t0 = time.time()
        net = train_network(X, y, arch, jax.random.PRNGKey(seed), name)
        emp = float(((np.asarray(T.h(jnp.asarray(X), net)) - y) ** 2).mean())
        print(f'  {label} restart {seed}: empirical risk {emp:.4e} ({time.time() - t0:.0f} s)', flush=True)
        if best is None or emp < best[0]:
            best = (emp, net)
    return best[1]


def train(d):
    z = np.load(os.path.join(DATA, f'train_domain_radial_{d}d.npz'))
    X, y = z['train_inputs'][:-N_VAL], z['train_labels'][:-N_VAL]   # the last N_VAL are held out
    print(f'd = {d}: {len(X)} points at n = {int(z["n_mc"]):.0e}, {N_STEPS} steps, batch {BATCH}, '
          f'cross-entropy, cosine decay from {LR0:g}, best of 3 restarts by empirical risk')
    net = best_of(X, y, (2 * d, 16, 16, 1), 3, 'loglambda', f'd = {d}')
    out = os.path.join(WEIGHTS, f'retrained_{d}d_16x2.npz')
    T.save_network(out, net)
    e, _ = errors(net, f'bench_test_{d}d')
    line('retrained network: test rms, max', f'{QUOTED[d]["rms"]}, {QUOTED[d]["mx"]}',
         f'{np.sqrt((e ** 2).mean()):.2e}, {np.abs(e).max():.2e}', f'(written to {out})')


def raw_inputs(X, rng):
    """(lambda, eta) -> the 8 free numbers of (A diag(lambda) A^T, A (3I - 2 diag(lambda)) A^T, A eta),
    A = Q diag(s) with Q a random rotation and log10 s uniform on (-1/4, 1/4); d = 2."""
    lam, m = X[:, :2], X[:, 2:]
    th = rng.uniform(0, 2 * np.pi, len(X))
    Q = np.stack([np.stack([np.cos(th), -np.sin(th)], -1), np.stack([np.sin(th), np.cos(th)], -1)], -2)
    A = Q * (10 ** rng.uniform(-0.25, 0.25, (len(X), 2)))[:, None, :]
    Sx = np.einsum('nij,nj,nkj->nik', A, lam, A)
    Sy = np.einsum('nij,nj,nkj->nik', A, 3.0 - 2.0 * lam, A)
    tri = lambda S: np.stack([S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]], -1)
    return np.column_stack([tri(Sx), tri(Sy), np.einsum('nij,nj->ni', A, m)])


def reduction():
    """Figure 2: the networks on the original arguments, same labels and budget, d = 2."""
    z = np.load(os.path.join(DATA, 'train_domain_radial_2d.npz'))
    b = np.load(os.path.join(DATA, 'bench_selection_2d.npz'))
    rng = np.random.default_rng(7)
    Xr, Xbr = raw_inputs(z['train_inputs'], rng), raw_inputs(b['X'], rng)
    y = z['train_labels']
    cached, _ = ladder_errors()
    os.makedirs(os.path.join(WEIGHTS, 'reduction'), exist_ok=True)
    print(f'd = 2: {len(Xr) - N_VAL} points, {N_STEPS} steps, cross-entropy, 2 restarts each; errors on the selection set')
    out = {}
    for width, depth in LADDER:
        net = best_of(Xr[:-N_VAL], y[:-N_VAL], (8,) + (width,) * depth + (1,), 2, 'identity', f'{width}x{depth}')
        T.save_network(os.path.join(WEIGHTS, 'reduction', f'raw_{width}x{depth}.npz'), net)
        e = np.asarray(T.h(jnp.asarray(Xbr), net)) - b['y']
        out[f'raw_{width}x{depth}'] = np.array([T.n_params(net), np.sqrt((e ** 2).mean()), np.abs(e).max()])
        line(f'original arguments {width}x{depth}: parameters, rms', f'{cached[(width, depth)][0]}, {cached[(width, depth)][1]:.1e}',
             f'{T.n_params(net)}, {np.sqrt((e ** 2).mean()):.2e}', f'(max {np.abs(e).max():.2e})')
    np.savez(os.path.join(DATA, 'reduction_gain_2d_recomputed.npz'), **out)


# ----------------------------------------------------------------------
SECTIONS = ['accuracy', 'timing', 'duotrio', 'map']


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--only', nargs='*', choices=SECTIONS, default=SECTIONS,
                    help='sections to run; --only alone runs none, for use with the flags')
    for flag in ('tail', 'train', 'reduction', 'map-mc', 'map-score', 'map-restarts'):
        ap.add_argument('--' + flag, action='store_true')
    ap.add_argument('--trials', metavar='PATH', help='npz with y, xref, xprobe: the trials of the MAP comparison')
    args = ap.parse_args()

    if 'accuracy' in args.only:
        print('\n=== accuracy of the networks (Section 3.2)')
        for d in (2, 3):
            accuracy(d)
    if 'timing' in args.only:
        print('\n=== evaluation time (Section 3.2, Figure 2)')
        timing()
    if 'duotrio' in args.only:
        print('\n=== duo-trio expansion (Section 4, Figure 4)')
        duotrio()
    if 'map' in args.only or args.map_mc or args.map_score or args.map_restarts:
        print('\n=== MAP comparison (Figure 3)')
        if args.trials is None:
            print('  skipped: the trials of hong2026 are not included; pass them with --trials PATH')
        else:
            map_comparison(load_trials(args.trials), rerun_mc=args.map_mc, rescore=args.map_score,
                           restarts=args.map_restarts)
    if args.tail:
        print('\n=== the ray that sets R (Section 3.2)')
        for d in (2, 3):
            tail(d)
    if args.train:
        print('\n=== retraining the networks')
        for d in (2, 3):
            train(d)
    if args.reduction:
        print('\n=== networks on the original arguments (Figure 2)')
        reduction()


if __name__ == '__main__':
    main()
