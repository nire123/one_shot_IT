# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:37:51 2026

@author: User
"""
import numpy as np

def merge_piecewise_linear_curves(all_knots, all_values):
    """
    Merge multiple piecewise linear curves by summing.
    
    Used to combine A-curve contributions from all source symbols.
    
    Parameters
    ----------
    all_knots : list of arrays
        Knot positions for each curve
    all_values : list of arrays
        Values for each curve
    
    Returns
    -------
    merged_knots : np.ndarray
        Merged knot positions (sorted, unique)
    merged_values : np.ndarray
        Merged values (sum of interpolated curves)
    """
    # Collect all unique knots
    merged_knots = np.unique(np.concatenate(all_knots))
    merged_knots = np.sort(merged_knots)
    
    # Interpolate each curve to merged knots and sum
    merged_values = np.zeros(len(merged_knots))
    for knots, values in zip(all_knots, all_values):
        merged_values += np.interp(merged_knots, knots, values)
    
    return merged_knots, merged_values


def _integrate_curve(knots, vals, f, num_refined_points=1000):
    """
    Integrate A_X(w) against kernel to get expected distortion.
    
    Formula (from proof):
        D_rand = ∫ A_X(w) · f(w) dw
    
    Parameters
    ----------
    w_knots : np.ndarray
        Knot positions
    A_vals : np.ndarray
        A_X(w) values at knots
    M : int
        Codebook size
    num_refined_points : int
        Number of points for grid refinement
    
    Returns
    -------
    float
        Expected distortion D_rand
    """    
        
    # Refine grid for accurate integration.
    # Always include the curve's own knots so the kernel is sampled where the
    # curve has breakpoints (critical when knots are concentrated near 0 and
    # the kernel decays rapidly, e.g. channel coding with large M or n).
    w_uniform = np.linspace(knots.min(), knots.max(), num_refined_points)
    # Add geometrically-spaced points near 0 to capture kernels that decay
    # with characteristic width ~1/M (large M means width << 1/num_refined_points).
    w_near_zero = np.geomspace(1e-9, max(knots[knots > 0].min() if np.any(knots > 0) else 0.1, 1e-9), num_refined_points)
    w_refined = np.clip(np.unique(np.concatenate([knots, w_uniform, w_near_zero])), 0.0, 1.0)
    A_refined = np.interp(w_refined, knots, vals)
    
    # Kernel: M(M-1)(1-w)^{M-2}
    kernel_refined = f(w_refined)
    
    # Integrand
    integrand_refined = A_refined * kernel_refined
    
    # Trapezoidal integration
    return float(np.trapezoid(integrand_refined, w_refined))


def integrate_curve_rd_exact(knots, vals, M, num_refined_points=1000):
    # D = M(M-1) ∫ A(w)(1-w)^{M-2} dw
    # Formula requires M >= 2: exponent M-2 < 0 for M < 2 makes kernel singular at w=1.
    if M < 2.0:
        return np.nan
    f = lambda w: M * (M - 1) * (1.0 - w) ** (M - 2)
    return _integrate_curve(knots, vals, f, num_refined_points=num_refined_points)


def integrate_curve_rd_exp_bound(knots, vals, M, num_refined_points=1000):
    """
    Upper bound on D using (1-w)^{M-1} <= exp(-(M-1)w).

    Applying this bound before IBP:
        D <= M * integral_0^1 exp(-(M-1)w) dA(w)
           = M(M-1) * integral_0^1 A(w) exp(-(M-1)w) dw  +  M * A(1) * exp(-(M-1))
    """
    if M < 2.0:
        return np.nan
    # Boundary term from IBP: M * A(1) * exp(-(M-1))
    boundary = M * float(vals[-1]) * np.exp(-(M - 1))
    f = lambda w: M * (M - 1) * np.exp(-(M - 1) * w)
    return _integrate_curve(knots, vals, f, num_refined_points=num_refined_points) + boundary


def integrate_curve_channel_coding_exact(knots, vals, M, num_refined_points=1000):
    # Pe = (M-1) ∫ (1-F(w))(1-w)^{M-2} dw
    if M < 2.0:
        return np.nan
    f = lambda w: (M - 1) * (1.0 - w) ** (M - 2)
    return _integrate_curve(knots, 1 - vals, f, num_refined_points=num_refined_points)


def integrate_curve_channel_coding_union_bound(knots, vals, M, num_refined_points=1000):
    """
    Upper bound on Pe using 1-(1-w)^{M-1} <= min(1, (M-1)w).

    After IBP:
        Pe <= (M-1) * integral_0^{1/(M-1)} (1-F(w)) dw
    """
    threshold = 1.0 / max(M - 1, 1e-15)
    f = lambda w: np.where(w <= threshold, float(M - 1), 0.0)
    return _integrate_curve(knots, 1 - vals, f, num_refined_points=num_refined_points)


def integrate_curve_jscc(knots, vals, M, num_refined_points=1000):
    """
    One-shot JSCC achievable bound (RCB):

        Pe <= M * integral_0^{1/M}  (1 - F(w))  dw

    The kernel is f(w) = M for w in [0, 1/M], zero elsewhere.

    Parameters
    ----------
    knots, vals : F-curve from OneShotJSCC.compute_curve  (both in [0, 1])
    M           : codebook size  (= |V| for blocklength 1)

    Returns
    -------
    float   clamped to [0, 1]
    """
    threshold = 1.0 / M
    f = lambda w: np.where(w <= threshold, float(M), 0.0)
    return min(1.0, _integrate_curve(knots, 1 - vals, f, num_refined_points=num_refined_points))




def compare_F_curves(knots1, F1, knots2, F2, label1="Curve 1", label2="Curve 2"):
    """
    Compare two F-curves by interpolating to common knots.
    
    Returns max absolute and relative differences.
    """
    # Merge knots
    all_knots = np.unique(np.concatenate([knots1, knots2]))
    all_knots = np.sort(all_knots)
    
    # Interpolate both curves
    F1_interp = np.interp(all_knots, knots1, F1)
    F2_interp = np.interp(all_knots, knots2, F2)
    
    # Compute differences
    abs_diff = np.abs(F1_interp - F2_interp)
    max_abs_diff = abs_diff.max()
    
    # Relative difference (where F > 0)
    mask = F1_interp > 1e-10
    if mask.any():
        rel_diff = abs_diff[mask] / F1_interp[mask]
        max_rel_diff = rel_diff.max()
    else:
        max_rel_diff = 0.0
    
    return {
        'max_abs_diff': max_abs_diff,
        'max_rel_diff': max_rel_diff,
        'all_close': np.allclose(F1_interp, F2_interp, atol=1e-10, rtol=1e-8)
    }


class F_curve:
    def __init__(self):
        
        self.all_knots = []
        self.all_values = []
    def add(self, k,v):
        self.all_knots.append(k)
        self.all_values.append(v)
    def finilize(self):
    
        self.merged_knots, self.merged_F = merge_piecewise_linear_curves (self.all_knots, self.all_values)
    
    def last_value(self):
        return self.merged_F[-1]

    def _integrate_curve(self, f, num_refined_points=1000):
        return _integrate_curve(self.merged_knots, self.merged_F, f, num_refined_points=num_refined_points)
    


    
