"""
G3 — exact random-coding bound vs the closed-form surrogate (channel & RD).

The random-coding probability has an exact kernel; in practice it is often
replaced by a looser closed form (union bound for channels, exponential bound
for rate-distortion). This figure quantifies that "kernel gap" at a fixed prior.
"""
import numpy as np
from _common import save, plt

from fbl import TypeBasedChannel, TypeBasedRD
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.channel_achievable_utils import z_channel
from fbl.F_curve import (
    integrate_curve_channel_coding_exact,
    integrate_curve_channel_coding_union_bound,
    integrate_curve_rd_exact,
    integrate_curve_rd_exp_bound,
)

N = 10
R_BITS = np.linspace(0.1, 0.85, 16)


def main():
    fig, (axc, axr) = plt.subplots(1, 2, figsize=(11, 4.3))

    # ---- channel: exact RC vs union bound ----
    tbc = TypeBasedChannel(z_channel(0.1), N)
    cc = tbc.compute_curve(memoryless_to_type_prior(np.array([0.55, 0.45]), N))
    Ms = [np.exp(N * Rb * np.log(2)) for Rb in R_BITS]
    ex = [integrate_curve_channel_coding_exact(*cc, M) for M in Ms]
    ub = [integrate_curve_channel_coding_union_bound(*cc, M) for M in Ms]
    axc.semilogy(R_BITS, ex, "o-", label="exact RC")
    axc.semilogy(R_BITS, ub, "s--", label="union bound")
    axc.set_xlabel("rate $R$ (bits/use)"); axc.set_ylabel(r"$P_e$")
    axc.set_title(f"Channel Z(0.1), $n={N}$"); axc.legend(fontsize=9)
    axc.grid(True, which="both", alpha=0.3)

    # ---- RD: exact RC vs exponential bound ----
    tbr = TypeBasedRD(np.array([0.5, 0.5]), np.array([[0.0, 1.0], [1.0, 0.0]]), N)
    cr = tbr.compute_curve(memoryless_to_type_prior(np.array([0.5, 0.5]), N))
    Msr = [np.exp(N * Rb * np.log(2)) for Rb in R_BITS]
    exr = [integrate_curve_rd_exact(*cr, M) for M in Msr]
    epr = [integrate_curve_rd_exp_bound(*cr, M) for M in Msr]
    axr.plot(R_BITS, exr, "o-", label="exact RC")
    axr.plot(R_BITS, epr, "s--", label="exponential bound")
    axr.set_xlabel("rate $R$ (bits/sym)"); axr.set_ylabel("expected distortion $D$")
    axr.set_title(f"RD (BMS+Hamming), $n={N}$"); axr.legend(fontsize=9)
    axr.grid(True, alpha=0.3)

    return save(fig, "ex3_bounds_vs_exact.png")


if __name__ == "__main__":
    main()
