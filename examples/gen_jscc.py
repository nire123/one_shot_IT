"""
JSCC -- error probability vs blocklength, for a reliable source/channel pair.

Pinned case: source BMS `P_V=[0.93,0.07]` (entropy H≈0.366 bit) through a
`BSC(0.05)` (capacity C≈0.714 bit).  Because **H < C** the source is transmissible:
the coded bounds (meta-converse and the RCU+ achievable bound at the optimal
conditional prior) decrease with `n`, while the **uncoded** scheme (send V through
the channel symbol-by-symbol, MAP-decode) degrades as `1-(1-p)^n`.

  single figure: P_e vs n with four curves
    - meta-converse        (lower bound, type-based, all n)
    - achievable (RCU+)    (upper bound at the Phi-view march optimum, all n)
    - Monte-Carlo          (realised error of random JSCC codes, small n only)
    - uncoded              (symbol-by-symbol baseline, all n)

For list size L=1 the codebook is pinned to M = |V|^n (no free rate knob).

Run:  python examples/gen_jscc.py  ->  examples/figures/jscc_error_vs_n.png
"""
import itertools
import numpy as np
from _common import save, plt

from fbl import TypeBasedJSCC
from fbl.one_shot_jscc import OneShotJSCC
from fbl.channel_achievable_utils import kronecker_power
from fbl.prioropt import phi_simplex as ps

P_V = np.array([0.93, 0.07])
DELTA = 0.05
W = np.array([[1 - DELTA, DELTA], [DELTA, 1 - DELTA]])    # BSC(0.05)
KV, KX = 2, 2
PREFIX = "jscc"


def _hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


H_V = float(-np.sum(P_V * np.log2(np.clip(P_V, 1e-12, None))))
CAP = float(1 - _hb(DELTA))


# ── per-n quantities ─────────────────────────────────────────────────────────
def converse_pe(n):
    tbj = TypeBasedJSCC(P_V, W, n)
    lb, _ = tbj.compute_converse(float(KV ** n))
    return lb


def achievable_pe(n):
    """RCU+ achievable bound at the Phi-view march optimum (P_e = 1 - J)."""
    M = float(KV ** n)
    res = ps.optimize(ps.build_program("jscc", P_V=P_V, W=W, n=n, M=M),
                      M, method="pgd", max_iter=4000, tol=1e-9)
    return 1.0 - res["J"]


def uncoded_pe(n):
    """Identity encoder x=v, per-symbol MAP; block error 1-(1-p_sym)^n."""
    scores = P_V[:, None] * W                  # (v, y)
    decoded = np.argmax(scores, axis=0)
    p_sym = sum(P_V[v] * W[v, decoded != v].sum() for v in range(KV))
    return 1.0 - (1.0 - p_sym) ** n


def _lift_encoder(Q1, n):
    Qn = np.zeros((KV ** n, KX ** n))
    for iv, vs in enumerate(itertools.product(range(KV), repeat=n)):
        for ix, xs in enumerate(itertools.product(range(KX), repeat=n)):
            Qn[iv, ix] = np.prod([Q1[vs[t], xs[t]] for t in range(n)])
    return Qn


def mc_pe(n, Q1, num_trials=200, seed=0):
    P_V_n = np.array([np.prod([P_V[i] for i in idx])
                      for idx in itertools.product(range(KV), repeat=n)])
    osj = OneShotJSCC(P_V_n, kronecker_power(W, n))
    return osj.mc(_lift_encoder(Q1, n), num_trials=num_trials, seed=seed)


# ── figure ───────────────────────────────────────────────────────────────────
def error_vs_n(n_bound=14, n_mc=7):
    from fbl.prioropt.achievability_jscc import AchievabilityJSCC
    Q1 = AchievabilityJSCC(P_V, W, 1).memoryless_optimal(KV)[1]   # single-letter enc.

    ns = np.arange(1, n_bound + 1)
    conv = np.array([converse_pe(n) for n in ns])
    ach = np.array([achievable_pe(n) for n in ns])
    unc = np.array([uncoded_pe(n) for n in ns])
    ns_mc = np.arange(1, n_mc + 1)
    mc = [mc_pe(n, Q1) for n in ns_mc]
    mc_mean = np.array([m["mean"] for m in mc])
    mc_std = np.array([m["std"] for m in mc])

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.semilogy(ns, unc, "D-", color="0.5", label="uncoded (symbol-by-symbol)")
    ax.semilogy(ns, ach, "s-", color="C1", label="achievable (RCU$^+$, optimal prior)")
    ax.errorbar(ns_mc, np.maximum(mc_mean, 1e-12), yerr=mc_std, fmt="o", color="C0",
                capsize=3, label="Monte-Carlo (random codes)")
    ax.semilogy(ns, np.maximum(conv, 1e-12), "k--", lw=1.5, label="meta-converse")
    ax.set_xlabel("blocklength $n$"); ax.set_ylabel(r"error probability $P_e$")
    ax.set_title(f"JSCC  BMS({P_V[1]}) over BSC({DELTA}):  "
                 f"$H={H_V:.3f} < C={CAP:.3f}$ bit/use")
    ax.legend(fontsize=8.5); ax.grid(True, which="both", alpha=0.3)
    print(f"  H(V)={H_V:.4f} bit, C={CAP:.4f} bit  (H<C: {H_V < CAP})")
    print(f"  achievable Pe: n=1 {ach[0]:.3e} -> n={ns[-1]} {ach[-1]:.3e}")
    print(f"  uncoded    Pe: n=1 {unc[0]:.3e} -> n={ns[-1]} {unc[-1]:.3e}")
    return save(fig, f"{PREFIX}_error_vs_n.png")


def main():
    print("[error_vs_n]"); error_vs_n()


if __name__ == "__main__":
    main()
