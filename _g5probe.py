import numpy as np, sys
sys.path.insert(0, "src")
from fbl import TypeBasedChannel
from fbl.channel_achievable_utils import z_channel
from fbl.type_based_utils import memoryless_to_type_prior, marginal_input
from fbl.prioropt import phi_view as pv

W = z_channel(0.1); KX = 2; LN2 = np.log(2.0)


def pe_exact(n, Q, M):
    return 1.0 - pv.J_typebased_channel(W, n, Q, M, "exact")


for n in (8, 20):
    print(f"=== n={n}: converse prior reused for achievability ===")
    print(f"{'Rb':>5} {'M':>8} {'conv_marg q1':>12} {'Pe_conv_full':>13} "
          f"{'Pe_conv_prod':>13} {'conv_LP_val':>11}")
    tbc = TypeBasedChannel(W, n)
    for Rb in np.linspace(0.06, 0.85, 12):
        M = float(np.exp(n * Rb * LN2))
        Q_conv, lp_val = tbc.optimize_prior(M)
        q1 = marginal_input(Q_conv, n, KX)[1]
        Q_cp = memoryless_to_type_prior(marginal_input(Q_conv, n, KX), n)
        pf = pe_exact(n, Q_conv, M)
        pp = pe_exact(n, Q_cp, M)
        print(f"{Rb:>5.2f} {M:>8.1f} {q1:>12.5f} {pf:>13.5e} {pp:>13.5e} {lp_val:>11.4e}",
              flush=True)
    print()
