"""
Utilities for One-Shot Achievable Bound
========================================

Helper functions for setting up sources, distortions, and priors.

Author: Nir
Date: 2026-04-02
"""

import numpy as np


def binary_memoryless_source(p, n):
    """
    Create distribution for n-length binary memoryless source.
    
    Parameters
    ----------
    p : float
        Probability of bit being 1
    n : int
        Sequence length
    
    Returns
    -------
    P_X : np.ndarray, shape (2^n,)
        Distribution over binary sequences
    """
    X_size = 2 ** n
    P_X_single = np.array([1.0 - p, p])
    
    P_X = np.zeros(X_size)
    
    for i in range(X_size):
        # Convert i to binary sequence
        seq = [(i >> bit) & 1 for bit in range(n)]
        # Probability is product of single-letter probabilities
        prob = np.prod([P_X_single[b] for b in seq])
        P_X[i] = prob
    
    return P_X


def hamming_distortion(n):
    """
    Create Hamming distortion matrix for binary sequences of length n.
    
    d[i,j] = number of bit positions where sequences i and j differ
    
    Parameters
    ----------
    n : int
        Sequence length
    
    Returns
    -------
    d : np.ndarray, shape (2^n, 2^n)
        Hamming distortion matrix
    """
    X_size = 2 ** n
    Y_size = 2 ** n
    
    d = np.zeros((X_size, Y_size))
    
    for i in range(X_size):
        for j in range(Y_size):
            # XOR gives 1 where bits differ
            xor = i ^ j
            # Count number of 1-bits
            dist = bin(xor).count('1')
            d[i, j] = float(dist)
    
    return d


def uniform_prior(n):
    """
    Create uniform prior over binary sequences of length n.
    
    Parameters
    ----------
    n : int
        Sequence length
    
    Returns
    -------
    Q : np.ndarray, shape (2^n,)
        Uniform distribution
    """
    size = 2 ** n
    return np.ones(size) / size


def setup_bms_hamming(p, n):
    """
    Set up BMS source with Hamming distortion.
    
    Convenience function for common test case.
    
    Parameters
    ----------
    p : float
        Crossover probability for BMS
    n : int
        Sequence length
    
    Returns
    -------
    P_X : np.ndarray
        Source distribution
    d : np.ndarray
        Hamming distortion matrix
    Q_Y : np.ndarray
        Uniform prior
    """
    P_X = binary_memoryless_source(p, n)
    d = hamming_distortion(n)
    Q_Y = uniform_prior(n)
    
    return P_X, d, Q_Y
