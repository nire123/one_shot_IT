"""
Utilities for Channel Coding Achievable Bound
==============================================

Helper functions for setting up channels and priors.

Author: Nir
Date: 2026-04-02
"""

import numpy as np


def binary_symmetric_channel(epsilon):
    """
    Create binary symmetric channel (BSC).
    
    W[0,0] = W[1,1] = 1-ε
    W[0,1] = W[1,0] = ε
    
    Parameters
    ----------
    epsilon : float
        Crossover probability
    
    Returns
    -------
    W : np.ndarray, shape (2, 2)
        BSC transition matrix
    """
    return np.array([[1 - epsilon, epsilon],
                     [epsilon, 1 - epsilon]])


def binary_erasure_channel(epsilon):
    """
    Create binary erasure channel (BEC).
    
    Outputs: {0, 1, e} where e is erasure
    
    Parameters
    ----------
    epsilon : float
        Erasure probability
    
    Returns
    -------
    W : np.ndarray, shape (2, 3)
        BEC transition matrix
    """
    return np.array([[1 - epsilon, 0, epsilon],
                     [0, 1 - epsilon, epsilon]])


def z_channel(epsilon):
    """
    Create Z-channel.
    
    W[0,0] = 1, W[0,1] = 0
    W[1,0] = ε, W[1,1] = 1-ε
    
    Parameters
    ----------
    epsilon : float
        Flip probability (1 → 0)
    
    Returns
    -------
    W : np.ndarray, shape (2, 2)
        Z-channel transition matrix
    """
    return np.array([[1.0, 0.0],
                     [epsilon, 1 - epsilon]])


def uniform_prior(X_size):
    """
    Create uniform prior over input alphabet.
    
    Parameters
    ----------
    X_size : int
        Input alphabet size
    
    Returns
    -------
    Q : np.ndarray, shape (X_size,)
        Uniform distribution
    """
    return np.ones(X_size) / X_size


def biased_prior(p):
    """
    Create biased binary prior.
    
    Q[0] = 1-p, Q[1] = p
    
    Parameters
    ----------
    p : float
        Probability of X=1
    
    Returns
    -------
    Q : np.ndarray, shape (2,)
        Binary distribution
    """
    return np.array([1.0 - p, p])


def setup_bsc_uniform(epsilon):
    """
    Set up BSC with uniform prior.
    
    Convenience function for common test case.
    
    Parameters
    ----------
    epsilon : float
        BSC crossover probability
    
    Returns
    -------
    W : np.ndarray, shape (2, 2)
        BSC transition matrix
    Q : np.ndarray, shape (2,)
        Uniform prior
    """
    W = binary_symmetric_channel(epsilon)
    Q = uniform_prior(2)
    return W, Q


def setup_z_channel_uniform(epsilon):
    """
    Set up Z-channel with uniform prior.
    
    Parameters
    ----------
    epsilon : float
        Z-channel flip probability
    
    Returns
    -------
    W : np.ndarray, shape (2, 2)
        Z-channel transition matrix
    Q : np.ndarray, shape (2,)
        Uniform prior
    """
    W = z_channel(epsilon)
    Q = uniform_prior(2)
    return W, Q


def kronecker_power(W, n):
    """
    Compute n-fold Kronecker product of channel W.
    
    W^⊗n represents the memoryless n-use extension of single-letter channel W.
    
    For a binary channel W (2x2), W^⊗n has size (2^n, 2^n).
    
    Parameters
    ----------
    W : np.ndarray, shape (X_size, Y_size)
        Single-letter channel transition matrix
    n : int
        Number of channel uses (blocklength)
    
    Returns
    -------
    W_n : np.ndarray, shape (X_size^n, Y_size^n)
        n-fold product channel
    
    Examples
    --------
    >>> W = z_channel(0.1)  # 2x2
    >>> W_3 = kronecker_power(W, 3)  # 8x8
    """
    W_n = W.copy()
    for _ in range(n - 1):
        W_n = np.kron(W_n, W)
    return W_n


def setup_z_channel_n_shot(epsilon, n):
    """
    Set up n-use Z-channel with uniform prior.
    
    Creates the n-fold product channel W^⊗n for the Z-channel.
    
    Parameters
    ----------
    epsilon : float
        Z-channel flip probability (single use)
    n : int
        Number of channel uses (blocklength)
    
    Returns
    -------
    W_n : np.ndarray, shape (2^n, 2^n)
        n-fold product Z-channel
    Q_n : np.ndarray, shape (2^n,)
        Uniform prior over n-bit sequences
    
    Examples
    --------
    >>> W, Q = setup_z_channel_n_shot(epsilon=0.1, n=5)
    >>> W.shape
    (32, 32)
    """
    W_single = z_channel(epsilon)
    W_n = kronecker_power(W_single, n)
    Q_n = uniform_prior(2**n)
    return W_n, Q_n
