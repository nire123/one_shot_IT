"""
Channel coding -- results for the pinned case  Z(0.1).

  G1  bound vs Monte-Carlo            (validation; small n only)
  G2  prior gap: optimal achievable prior vs optimal memoryless vs
      marginal-memoryless (achievable / converse)            -- the centerpiece
  G3  exact random-coding error vs the union-bound surrogate
  G4  error spectrum: achievability-optimal vs converse-optimal prior

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


# ── G5 ─ full optimal prior vs its product (marginalized) version ─────────────
def g5_full_vs_product(n):
    """Take each optimal prior (converse / achievable) and its i.i.d. *product*
    version (per-symbol marginal applied i.i.d.); show how the exact achievable
    bound changes. The converse prior is poor for achievability (full) but its
    product recovers most of it; the achievable prior is barely changed."""
    tbc = TypeBasedChannel(W, n)
    prog = ps.build_program("channel", W=W, n=n, kernel="exact")
    R_bits = np.linspace(0.06, 0.85, 12)
    cf, cp_, af, ap = [], [], [], []
    warm = None
    for Rb in R_bits:
        M = float(np.exp(n * Rb * LN2))
        res = ps.optimize(prog, M, method="pgd", max_iter=3000, tol=1e-7, warm_start=warm)
        warm = res["Q"]
        Q_ach, (Q_conv, _) = res["Q"], tbc.optimize_prior(M)
        Q_ap = memoryless_to_type_prior(marginal_input(Q_ach, n, KX), n)
        Q_cp = memoryless_to_type_prior(marginal_input(Q_conv, n, KX), n)
        af.append(_pe_exact(tbc, n, Q_ach, M)); ap.append(_pe_exact(tbc, n, Q_ap, M))
        cf.append(_pe_exact(tbc, n, Q_conv, M)); cp_.append(_pe_exact(tbc, n, Q_cp, M))
    A = lambda v: np.array(v)
    cf, cp_, af, ap = map(A, (cf, cp_, af, ap))

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.semilogy(R_bits, cf, "s-", color="C3", label="converse-optimal — full")
    ax.semilogy(R_bits, cp_, "v--", color="C3", label="converse-optimal — product")
    ax.semilogy(R_bits, af, "o-", color="C1", label="achievability-optimal — full")
    ax.semilogy(R_bits, ap, "^--", color="C1", label="achievability-optimal — product")
    ax.set_xlabel("rate $R$ (bits/use)"); ax.set_ylabel(r"achievable $P_e$ (exact RC)")
    ax.set_title(f"G5  {TITLE}, $n={n}$: full optimal prior vs its product version")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    print(f"  G5 n={n}: converse full/product ratio (mid) "
          f"{cf[len(cf)//2] / cp_[len(cp_)//2]:.1f}x, "
          f"achiev {af[len(af)//2] / ap[len(ap)//2]:.3f}x")
    return save(fig, f"{PREFIX}_g5_full_vs_product_n{n}.png")


def main():
    print("[g1]"); g1_mc_spread(8)
    for n in (8, 20):
        print(f"[g2 n={n}]"); g2_prior_gap(n)
        print(f"[g3 n={n}]"); g3_bounds_vs_exact(n)
        print(f"[g4 n={n}]"); g4_fcurve_compare(n)
        print(f"[g5 n={n}]"); g5_full_vs_product(n)


if __name__ == "__main__":
    main()
