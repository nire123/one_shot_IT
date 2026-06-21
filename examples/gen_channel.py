"""
Channel coding -- results for the pinned case  Z(0.1).

  G1  bound vs Monte-Carlo            (validation; small n only)
  G2  prior gap: optimal achievable prior vs optimal memoryless vs
      marginal-memoryless (achievable / converse)            -- the centerpiece
  G3  exact random-coding error vs the union-bound surrogate
  G4  error spectrum: achievability-optimal vs converse-optimal prior
  G5  optimal achievable prior vs its i.i.d. product (marginalization cost)
  rate_vs_n  fix the error eps; achievable & converse rate vs blocklength

The achievability-optimal prior is the Phi-view simplex march (KKT/FW-gap
certified); the headline bound is the EXACT random-coding kernel
(Pe = 1 - c^T Phi(A Q), Phi(s)=(1-(1-s)^M)/M, M=e^{nR}).  G1 stays at n=8
(it only validates the bound); G2/G3/G4 run at n=8 and n=20.

Run:  python examples/gen_channel.py  ->  examples/figures/channel_*.png
"""
import numpy as np
from _common import save, plt

from fbl import OneShotChannel, TypeBasedChannel
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.channel_achievable_utils import z_channel, kronecker_power
from fbl.F_curve import (integrate_curve_channel_coding_exact,
                         integrate_curve_channel_coding_union_bound)
from fbl.type_based_utils import marginal_input
from fbl.prioropt import phi_simplex as ps
from fbl.prioropt import phi_view as pv

W = z_channel(0.1)
KX = W.shape[0]
PREFIX = "channel"
TITLE = "channel Z(0.1)"
LN2 = np.log(2.0)


# -- helpers ------------------------------------------------------------------
def _pe_exact(tbc, n, Q_type, M):
    """P_e of a type prior under the exact RC kernel (= 1 - c^T Phi(A Q))."""
    return 1.0 - pv.J_typebased_channel(W, n, Q_type, M, "exact")


def _best_memoryless(n, M, grid):
    """Best i.i.d. prior under the exact kernel: min over single-letter bias."""
    return min(_pe_from_mem(n, np.array([1 - q, q]), M) for q in grid)


def _pe_from_mem(n, q1, M):
    return 1.0 - pv.J_typebased_channel(W, n, memoryless_to_type_prior(q1, n), M, "exact")


# ── G1 ─ bound vs Monte-Carlo (validation, n=8) ──────────────────────────────
def g1_mc_spread(n=8):
    osc = OneShotChannel(kronecker_power(W, n))
    q1 = np.array([0.6, 0.4])
    Q = np.array([q1[1] ** bin(x).count("1") * q1[0] ** (n - bin(x).count("1"))
                  for x in range(2 ** n)])
    Q /= Q.sum()
    curve = osc.compute_curve(Q)
    R_bits = np.linspace(0.12, 0.82, 12)
    Ms = [max(2, int(round(np.exp(n * Rb * LN2)))) for Rb in R_bits]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _ in range(60):
        pe = [osc.evaluate(osc.draw_random_code(Q, M, rng)) for M in Ms]
        ax.plot(R_bits, pe, color="C0", alpha=0.10, lw=0.8)
    ax.plot([], [], color="C0", alpha=0.4, label="60 random codebooks")
    ax.plot(R_bits, [osc.theory(curve, M) for M in Ms], "k-", lw=2.2,
            label="RC expectation (exact)")
    ax.set_xlabel("rate $R$ (bits/use)"); ax.set_ylabel(r"$P_e$")
    ax.set_title(f"G1  {TITLE}, $n={n}$: bound vs Monte-Carlo")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return save(fig, f"{PREFIX}_g1_mc_spread.png")


# ── G2 ─ prior gap: optimal vs memoryless variants (exact kernel) ────────────
def g2_prior_gap(n):
    tbc = TypeBasedChannel(W, n)
    prog = ps.build_program("channel", W=W, n=n, kernel="exact")
    R_bits = np.linspace(0.06, 0.85, 12)
    mgrid = np.linspace(0.02, 0.98, 33)

    pe_ach, pe_ml, pe_am, pe_cm, pe_conv = [], [], [], [], []
    warm = None
    for Rb in R_bits:
        M = float(np.exp(n * Rb * LN2))
        res = ps.optimize(prog, M, method="pgd", max_iter=3000, tol=1e-7,
                          warm_start=warm)
        warm = res["Q"]                                   # warm-start next rate
        Q_ach = res["Q"]
        Q_conv, _ = tbc.optimize_prior(M)
        Q_am = memoryless_to_type_prior(marginal_input(Q_ach, n, KX), n)
        Q_cm = memoryless_to_type_prior(marginal_input(Q_conv, n, KX), n)
        pe_ach.append(_pe_exact(tbc, n, Q_ach, M))
        pe_ml.append(_best_memoryless(n, M, mgrid))
        pe_am.append(_pe_exact(tbc, n, Q_am, M))
        pe_cm.append(_pe_exact(tbc, n, Q_cm, M))
        pe_conv.append(tbc.optimize_prior(M)[1])
    A = lambda v: np.array(v)
    pe_ach, pe_ml, pe_am, pe_cm, pe_conv = map(A, (pe_ach, pe_ml, pe_am, pe_cm, pe_conv))
    gain = (pe_ml - pe_ach) / pe_ml * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax.semilogy(R_bits, pe_ml, "o-", color="C0", label="optimal memoryless")
    ax.semilogy(R_bits, pe_am, "v--", color="C2", label="marginal memoryless (achiev.)")
    ax.semilogy(R_bits, pe_cm, "^:", color="C3", label="marginal memoryless (converse)")
    ax.semilogy(R_bits, pe_ach, "s-", color="C1", label="optimal achievable prior")
    ax.semilogy(R_bits, pe_conv, "k-", lw=1, alpha=0.6, label="meta-converse")
    ax.set_xlabel("rate $R$ (bits/use)"); ax.set_ylabel(r"$P_e$ (exact RC)")
    ax.set_title(f"G2  {TITLE}, $n={n}$: prior gap"); ax.legend(fontsize=7.5)
    ax.grid(True, which="both", alpha=0.3)
    ax2.plot(R_bits, gain, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/use)")
    ax2.set_ylabel("gain of optimal over best memoryless (%)")
    ax2.set_title("non-product prior gain"); ax2.grid(True, alpha=0.3)
    print(f"  G2 n={n}: max gain {gain.max():.2f}% at R={R_bits[gain.argmax()]:.2f}b")
    return save(fig, f"{PREFIX}_g2_prior_gap_n{n}.png")


# ── G3 ─ exact RC vs union bound ─────────────────────────────────────────────
def g3_bounds_vs_exact(n):
    tbc = TypeBasedChannel(W, n)
    cc = tbc.compute_curve(memoryless_to_type_prior(np.array([0.55, 0.45]), n))
    R_bits = np.linspace(0.1, 0.85, 16)
    Ms = [np.exp(n * Rb * LN2) for Rb in R_bits]
    ex = [integrate_curve_channel_coding_exact(*cc, M) for M in Ms]
    ub = [integrate_curve_channel_coding_union_bound(*cc, M) for M in Ms]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(R_bits, ex, "o-", label="exact RC")
    ax.semilogy(R_bits, ub, "s--", label="union bound")
    ax.set_xlabel("rate $R$ (bits/use)"); ax.set_ylabel(r"$P_e$")
    ax.set_title(f"G3  {TITLE}, $n={n}$: bound vs exact RC")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    return save(fig, f"{PREFIX}_g3_bounds_vs_exact_n{n}.png")


# ── G4 ─ error spectrum: achievability- vs converse-optimal prior ────────────
def g4_fcurve_compare(n):
    Rb = 0.25
    R = n * Rb * LN2; M = float(np.exp(R)); w0 = 1.0 / M
    tbc = TypeBasedChannel(W, n)
    prog = ps.build_program("channel", W=W, n=n, kernel="exact")
    P_conv, _ = tbc.optimize_prior(M)
    P_ach = ps.optimize(prog, M, method="pgd", max_iter=4000, tol=1e-8)["Q"]

    kc, Fc = tbc.compute_curve(P_conv)
    ka, Fa = tbc.compute_curve(P_ach)
    bnd_c = integrate_curve_channel_coding_exact(kc, Fc, M)
    bnd_a = integrate_curve_channel_coding_exact(ka, Fa, M)

    def espec(k, F, w):
        return np.clip(1.0 - np.interp(w, k, F), 1e-12, None)
    z = np.linspace(0.45 * R, 2.6 * R, 400); w = np.exp(-z)

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.axvspan(R, z.max(), color="0.92", label=r"achievability integrates here ($z\geq R$)")
    ax.semilogy(z, espec(kc, Fc, w), color="C0", lw=2,
                label=f"converse-optimal prior  (exact $P_e$={bnd_c:.1e})")
    ax.semilogy(z, espec(ka, Fa, w), color="C1", lw=2,
                label=f"achievability-optimal prior  (exact $P_e$={bnd_a:.1e})")
    ax.axvline(R, color="k", ls=":", lw=1.2)
    ax.set_xlabel("z = -log(PEP)  (nats)")
    ax.set_ylabel("error spectrum  Pr[-log PEP <= z]")
    ax.set_title(f"G4  {TITLE}, $n={n}$, $R={Rb}$b: achievable vs converse prior")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    print(f"  G4 n={n}: converse Pe={bnd_c:.2e}, achievable Pe={bnd_a:.2e}")
    return save(fig, f"{PREFIX}_g4_fcurve_compare_n{n}.png")


# ── G5 ─ optimal achievable prior: full vs its i.i.d. product ─────────────────
def g5_full_vs_product(n):
    """How the exact achievable bound changes when the (well-defined,
    KKT-certified) optimal achievable prior is replaced by its i.i.d. *product*
    version (per-symbol marginal applied i.i.d.). Left: the two bounds; right: the
    marginalization cost. (The converse prior is omitted here: its single-threshold
    optimum is non-unique where the converse is slack, so reusing it for the
    achievability integral is ill-defined.)"""
    tbc = TypeBasedChannel(W, n)
    prog = ps.build_program("channel", W=W, n=n, kernel="exact")
    R_bits = np.linspace(0.06, 0.85, 12)
    af, ap = [], []
    warm = None
    for Rb in R_bits:
        M = float(np.exp(n * Rb * LN2))
        res = ps.optimize(prog, M, method="pgd", max_iter=3000, tol=1e-7, warm_start=warm)
        warm = res["Q"]
        Q_ap = memoryless_to_type_prior(marginal_input(res["Q"], n, KX), n)
        af.append(1.0 - res["J"]); ap.append(_pe_exact(tbc, n, Q_ap, M))
    af, ap = np.array(af), np.array(ap)
    cost = (ap - af) / af * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax.semilogy(R_bits, af, "o-", color="C1", label="optimal achievable prior (full)")
    ax.semilogy(R_bits, ap, "^--", color="C2", label="its i.i.d. product (marginalized)")
    ax.set_xlabel("rate $R$ (bits/use)"); ax.set_ylabel(r"achievable $P_e$ (exact RC)")
    ax.set_title(f"G5  {TITLE}, $n={n}$: optimal prior vs its product")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    ax2.plot(R_bits, cost, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/use)"); ax2.set_ylabel("marginalization cost (%)")
    ax2.set_title("cost of the i.i.d. product"); ax2.grid(True, alpha=0.3)
    print(f"  G5 n={n}: max marginalization cost {cost.max():.2f}%")
    return save(fig, f"{PREFIX}_g5_full_vs_product_n{n}.png")


# ── rate vs blocklength at fixed error ────────────────────────────────────────
def _capacity_bits():
    """Capacity of W (bits/use) = max_q I(X;Y), 1-D search over q = P(X=1)."""
    best = 0.0
    for q in np.linspace(1e-4, 1 - 1e-4, 400):
        Px = np.array([1 - q, q]); Py = Px @ W
        I = sum(Px[x] * W[x, y] * np.log2(W[x, y] / Py[y])
                for x in range(2) for y in range(2) if W[x, y] > 0 and Py[y] > 0)
        best = max(best, I)
    return best


def _rate_at_eps(err_of_Rb, eps, R_hi, iters=16):
    """Largest rate (bits) with err <= eps, by bisection (err monotone in rate)."""
    lo, hi = 0.0, R_hi
    if err_of_Rb(hi) <= eps:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if err_of_Rb(mid) <= eps else (lo, mid)
    return lo


def rate_vs_n(eps=1e-3, n_list=(4, 6, 8, 10, 12, 14, 16, 18, 20, 22)):
    """Fix the error probability eps; plot achievable (Phi-march) and converse
    (meta-converse LP) rate vs blocklength, converging toward capacity."""
    C = _capacity_bits()
    R_hi = C + 0.35
    R_ach, R_conv = [], []
    for n in n_list:
        prog = ps.build_program("channel", W=W, n=n, kernel="exact")
        warm = {"Q": None}

        def err_ach(Rb):
            res = ps.optimize(prog, float(np.exp(n * Rb * LN2)), method="pgd",
                              max_iter=2500, tol=1e-7, warm_start=warm["Q"])
            warm["Q"] = res["Q"]
            return 1.0 - res["J"]

        tbc = TypeBasedChannel(W, n)
        err_conv = lambda Rb: tbc.optimize_prior(float(np.exp(n * Rb * LN2)))[1]
        R_ach.append(_rate_at_eps(err_ach, eps, R_hi))
        R_conv.append(_rate_at_eps(err_conv, eps, R_hi))
        print(f"  n={n}: R_ach={R_ach[-1]:.4f}  R_conv={R_conv[-1]:.4f}  (C={C:.4f})",
              flush=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(C, color="k", ls=":", lw=1.3, label=f"capacity $C={C:.3f}$")
    ax.plot(n_list, R_conv, "^-", color="C0", label=r"converse (max rate at $P_e=\epsilon$)")
    ax.plot(n_list, R_ach, "s-", color="C1", label=r"achievable ($\Phi$-march at $P_e=\epsilon$)")
    ax.set_xlabel("blocklength $n$"); ax.set_ylabel("rate $R$ (bits/use)")
    ax.set_title(f"{TITLE}: rate vs blocklength at fixed $P_e=\\epsilon={eps:g}$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return save(fig, f"{PREFIX}_rate_vs_n.png")


def main():
    print("[g1]"); g1_mc_spread(8)
    for n in (8, 20):
        print(f"[g2 n={n}]"); g2_prior_gap(n)
        print(f"[g3 n={n}]"); g3_bounds_vs_exact(n)
        print(f"[g4 n={n}]"); g4_fcurve_compare(n)
        print(f"[g5 n={n}]"); g5_full_vs_product(n)
    print("[rate_vs_n]"); rate_vs_n(eps=1e-3)


if __name__ == "__main__":
    main()
