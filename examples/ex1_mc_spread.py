"""
G1 — Monte-Carlo spread vs the RCU expectation (channel coding).

The achievable bound is an *expectation* over the random codebook ensemble. This
figure draws many actual codebooks (in the lifted X^n space, exact ML decoding)
and overlays their realised error probabilities on the analytic RCU expectation,
to show how much a single random code deviates from the mean.
"""
import numpy as np
from _common import save, plt

from fbl import OneShotChannel
from fbl.channel_achievable_utils import z_channel, kronecker_power

N = 6
W = z_channel(0.1)
N_CODES = 60
R_BITS = np.linspace(0.12, 0.78, 12)
Q1 = np.array([0.6, 0.4])           # single-letter input prior (i.i.d. ensemble)


def product_prior(q1, n):
    """i.i.d. prior over X^n from single-letter q1 (X={0,1})."""
    P = np.empty(2 ** n)
    for x in range(2 ** n):
        ones = bin(x).count("1")
        P[x] = q1[1] ** ones * q1[0] ** (n - ones)
    return P / P.sum()


def main():
    osc = OneShotChannel(kronecker_power(W, N))
    Q = product_prior(Q1, N)
    curve = osc.compute_curve(Q)
    Ms = [max(2, int(round(np.exp(N * Rb * np.log(2))))) for Rb in R_BITS]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _ in range(N_CODES):
        pe = [osc.evaluate(osc.draw_random_code(Q, M, rng)) for M in Ms]
        ax.plot(R_BITS, pe, color="C0", alpha=0.10, lw=0.8)
    ax.plot([], [], color="C0", alpha=0.4, lw=1, label=f"{N_CODES} random codebooks")
    exp = [osc.theory(curve, M) for M in Ms]
    ax.plot(R_BITS, exp, "k-", lw=2.2, label="RCU expectation (exact RC)")
    ax.set_xlabel("rate $R$ (bits/use)")
    ax.set_ylabel(r"$P_e$")
    ax.set_title(f"Z(0.1), $n={N}$: random-code spread around the expectation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return save(fig, "ex1_mc_spread.png")


if __name__ == "__main__":
    main()
