# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 00:30:09 2026

@author: User
"""

# minimal_indexers_numpy.py
"""
Minimal numpy-based implementation of two core indexing algorithms:
1. composition_to_index: composition -> integer
2. matrix_to_index: matrix with fixed row sums -> integer
"""

import numpy as np
import math
from functools import lru_cache
import itertools
from scipy.special import gammaln

@lru_cache(maxsize=10000)
def binom(n: int, k: int) -> int:
    """Binomial coefficient C(n,k) with caching"""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    return math.comb(n, k)

####################################################
# Type class: enumeration, indexing, sizes
####################################################

def enumerate_type_class(n: int, k: int):
    if k == 1:
        # Base case: single part must contain all of n
        yield np.array([n], dtype=int)
    else:
        # Try all possible values for the first position
        for first in range(n + 1):
            # Recursively generate the rest
            for rest in enumerate_type_class(n - first, k - 1):
                # Prepend first value to rest
                comp = np.empty(k, dtype=int)
                comp[0] = first
                comp[1:] = rest
                yield comp

def composition_to_index(comp: np.ndarray) -> int:
    n = int(np.sum(comp))
    k = len(comp)

    index = 0
    remaining_n = n
    remaining_k = k

    for val in comp:
        val = int(val)
        if val > 0 and remaining_k > 1:
            # Hockey-stick identity: O(1) instead of O(n)
            # sum_{v=0}^{val-1} C(r-v+m-2, m-2) = C(r+m-1, m-1) - C(r-val+m-1, m-1)
            r = remaining_n
            m = remaining_k

            total = binom(r + m - 1, m - 1)
            after = binom(r - val + m - 1, m - 1)
            index += total - after

        remaining_n -= val
        remaining_k -= 1

    return index


def batch_composition_to_index(comps: np.ndarray) -> np.ndarray:
    """
    Vectorized composition_to_index for a batch of compositions.

    Parameters
    ----------
    comps : int array, shape (N, k)  — each row sums to the same n

    Returns
    -------
    indices : int64 array, shape (N,)
    """
    N, k = comps.shape
    n    = int(comps[0].sum())

    # Precompute binom(r + m - 1, m - 1) for all needed (r, m) pairs.
    # r ranges in [0..n], m in [1..k].  Store in table[r, m] = C(r+m-1, m-1).
    binom_table = np.zeros((n + 1, k + 1), dtype=np.int64)
    for r in range(n + 1):
        for m in range(1, k + 1):
            binom_table[r, m] = binom(r + m - 1, m - 1)

    indices      = np.zeros(N, dtype=np.int64)
    remaining_n  = comps.sum(axis=1).copy()   # (N,) all start at n
    remaining_k  = np.full(N, k, dtype=np.int64)

    for col in range(k):
        vals = comps[:, col].astype(np.int64)
        mask = (vals > 0) & (remaining_k > 1)
        if mask.any():
            r = remaining_n[mask]
            m = remaining_k[mask]
            v = vals[mask]
            total = binom_table[r, m]
            after = binom_table[np.clip(r - v, 0, n), m]
            indices[mask] += total - after
        remaining_n  -= vals
        remaining_k  -= 1

    return indices


def index_to_composition(index: int, n: int, k: int) -> np.ndarray:
    comp = np.zeros(k, dtype=int)
    remaining_n = n
    remaining_k = k
    remaining_index = index
    
    for pos in range(k):
        if remaining_k == 1:
            comp[pos] = remaining_n
            break
        
        r = remaining_n
        m = remaining_k
        total = binom(r + m - 1, m - 1)
        
        # Binary search for value at this position
        lo, hi = 0, remaining_n
        
        while lo < hi:
            mid = (lo + hi + 1) // 2
            after = binom(r - mid + m - 1, m - 1)
            count_up_to_mid = total - after
            
            if count_up_to_mid <= remaining_index:
                lo = mid
            else:
                hi = mid - 1
        
        val = lo
        
        # Update remaining_index
        if val > 0:
            after = binom(r - val + m - 1, m - 1)
            remaining_index -= (total - after)
        
        comp[pos] = val
        remaining_n -= val
        remaining_k -= 1
    
    return comp


def log_size_type_class(A):
    
    n = A.sum()
    return gammaln(n + 1.0) - np.sum(gammaln(A + 1.0))

def composition_count(n: int, k: int) -> int:
    return binom(n + k - 1, k - 1)
        

####################################################
# Conditional type class
####################################################

def log_size_conditional_type_class(A):
    
    
    A = np.asarray(A, dtype=np.int64)
    T_y = A.sum(axis=tuple(range(1, A.ndim)))

    log_size = 0.0

    # sum_y log(T_y!)
    log_size += np.sum(gammaln(T_y + 1.0))

    # subtract sum_{y,x} log(A[y,x]!)
    log_size -= np.sum(gammaln(A + 1.0))

    return log_size


def conditional_composition_count(t_y, size_x):
    size = 1
    for y in range(len(t_y)):
        if t_y[y] > 0:
            size *= composition_count(t_y[y], size_x)
    return size


def matrix_to_index(matrix: np.ndarray) -> int:
    n_rows, n_cols = matrix.shape
    row_sums = matrix.sum(axis=1)
    
    # Precompute row counts for mixed-radix
    row_counts = np.array([composition_count(int(rs), n_cols) for rs in row_sums])
    
    index = 0
    multiplier = 1
    
    # Process rows from bottom to top
    for i in range(n_rows - 1, -1, -1):
        row_index = composition_to_index(matrix[i])
        index += row_index * multiplier
        multiplier *= row_counts[i]
    
    return index


def index_to_matrix(index: int, row_sums: np.ndarray, n_cols: int) -> np.ndarray:
    n_rows = len(row_sums)
    
    # Precompute row counts
    row_counts = np.array([composition_count(int(rs), n_cols) for rs in row_sums])
    
    matrix = np.zeros((n_rows, n_cols), dtype=int)
    remaining = index
    
    # Process rows from bottom to top
    for i in range(n_rows - 1, -1, -1):
        row_index = remaining % row_counts[i]
        matrix[i] = index_to_composition(row_index, int(row_sums[i]), n_cols)
        remaining //= row_counts[i]
    
    return matrix

     
def enumerate_conditional_type_class(t_y: np.ndarray, x_size: int):
    y_size = len(t_y)
    
    # Find active y values (those with non-zero counts)
    active_ys = [(y, int(t_y[y])) for y in range(y_size) if t_y[y] > 0]
    
    if len(active_ys) == 0:
        # All zeros - only one possibility
        yield np.zeros((y_size, x_size), dtype=np.int64)
        return
    
    # Create list of generators for each active y
    generators = []
    for y, n_y in active_ys:
        gen = list(enumerate_type_class(n_y, x_size))
        generators.append(gen)
    
    # Enumerate Cartesian product
    for compositions in itertools.product(*generators):
        t_xy = np.zeros((y_size, x_size), dtype=np.int64)
        for (y, n_y), comp in zip(active_ys, compositions):
            t_xy[y, :] = comp
        yield t_xy
     
            
####################################################
# Tests
####################################################

def test_conditional_type_enumaration(n = 5, k_v = 3, k_y = 4):
             
    print(f"Test conditional type enumaration: n={n}; k_y={k_y}; k_v={k_v}")
    num_of_element_in_type_class = 0
    
    for i_T_y, T_y in enumerate(enumerate_type_class(n, k_y)):
        assert i_T_y == composition_to_index(T_y)
        assert (index_to_composition(i_T_y, n, k_y) == T_y).all()
        
        num_of_element_in_type_class += np.exp(log_size_type_class(T_y))
                
        num_of_element_in_conditional_type_class = 0
        for i_T_v_given_y, T_yv in enumerate(enumerate_conditional_type_class(T_y, k_v)):
            assert i_T_v_given_y == matrix_to_index(T_yv)
            assert ((index_to_matrix(i_T_v_given_y, T_y, k_v) == T_yv).all())
            num_of_element_in_conditional_type_class += np.exp(log_size_conditional_type_class(T_yv))
            
        assert round(num_of_element_in_conditional_type_class) == k_v**n
        assert i_T_v_given_y + 1 == conditional_composition_count(T_y, k_v)
    
    assert i_T_y+1 == composition_count(n, k_y)    
    assert round(num_of_element_in_type_class) == k_y**n


def test_conditional_type_enumaration2(n = 5, k_v = 3, k_x = 3, k_y = 4):

        
    print(f"Test conditional type enumaration 2: n={n}; k_y={k_y}; k_x={k_x}; k_v={k_v}")
    for i_T_y, T_y in enumerate(enumerate_type_class(n, k_y)):
        assert i_T_y == composition_to_index(T_y)
        assert (index_to_composition(i_T_y, n, k_y) == T_y).all()
        for i_T_v_given_y, T_yv in enumerate(enumerate_conditional_type_class(T_y, k_v)):
            assert i_T_v_given_y == matrix_to_index(T_yv)
            assert ((index_to_matrix(i_T_v_given_y, T_y, k_v) == T_yv).all())
            for i_T_x_given_yv, T_yvx in enumerate(enumerate_conditional_type_class(T_yv.flatten(), k_x)):
                assert i_T_x_given_yv == matrix_to_index(T_yvx)
                assert ((index_to_matrix(i_T_x_given_yv, T_yv.flatten(), k_x) == T_yvx).all())


def compute_conditional_2_joint(n, x_size, v_size):
    d1 = dict()
    for i_T_vx, T_vx in enumerate(enumerate_type_class(n, x_size*v_size)):
        T_vx = T_vx.reshape((v_size, x_size))
        
        T_v = T_vx.sum(axis=1)
        i_T_v = composition_to_index(T_v)
        i_T_x_given_v = int(matrix_to_index(T_vx))
        d1[i_T_vx] = (i_T_v, i_T_x_given_v)
    return d1


def test_conditional_joint_type(n, v_size, x_size):
    print(f'test_conditional_joint_type: n={n}; (v,x)=({v_size},{x_size})')
    d1 = compute_conditional_2_joint(n, x_size, v_size)    
    assert len(set(d1.values())) == len(d1)
    
    
    for i_T_v, T_v in enumerate(enumerate_type_class(n, v_size)):
        for i_T_x_given_v, T_vx in enumerate(enumerate_conditional_type_class(T_v, x_size)):
            i_T_vx = composition_to_index(T_vx.flatten())
            assert d1[i_T_vx][0] == i_T_v
            assert d1[i_T_vx][1] == i_T_x_given_v
            # print(d1[i_T_vx], i_T_vx)



if __name__ == "__main__" and False:
    
    
    for n in range(1, 7):            
        test_conditional_joint_type(n, v_size=3, x_size=4)
    for n in range(1, 7):            
        test_conditional_joint_type(n, v_size=2, x_size=3)


    for n in range(2,10):
        test_conditional_type_enumaration(n = n, k_v = 2, k_y = 2)
    
    for n in range(2,6):
        test_conditional_type_enumaration(n = n, k_v = 3, k_y = 4)
    
    for n in range(2,5):
        test_conditional_type_enumaration(n = n, k_v = 4, k_y = 5)
        
    for n in range(2,4):
        test_conditional_type_enumaration(n = n, k_v = 5, k_y = 6)
        

    for n in range(2,8):
        test_conditional_type_enumaration2(n = n, k_v = 2, k_x = 2, k_y = 2)

    for n in range(2,5):
        test_conditional_type_enumaration2(n = n, k_v = 3, k_x = 3, k_y = 4)
    






class conditional_enum:
    def __init__(self, n, k_y, k_x):
        self.n = n
        self.k_y = k_y
        self.k_x = k_x
        
        self.len = composition_count(n, k_y*k_x) # number of joint types
        self.len_y = composition_count(n, k_y)   # number of primarily types
        
        self.joint_ix_2_prime = -np.ones(self.len, dtype=np.int32)
        self.joint_ix_2_conditional = -np.ones(self.len, dtype=np.int32)

        # ToDo: do a single loop here
        self.len_cond_given_prime = np.array([ conditional_composition_count(T_y, k_x) for  T_y in enumerate_type_class(n, k_y) ])
        self.size_cond_given_prime = np.array([ log_size_type_class(T_y) for  T_y in enumerate_type_class(n, k_y) ])
        self.size_prime = np.array([ log_size_type_class(T_x) for  T_x in enumerate_type_class(n, k_x) ])
        
        assert self.len_cond_given_prime.sum() == self.len

        self.prime_ix_2_cond_start_ix = np.insert(np.cumsum(self.len_cond_given_prime),0,0)
            
    def tuple_2_ix(self, i_T_y, i_T_x_given_y):
        return self.prime_ix_2_cond_start_ix[i_T_y]+i_T_x_given_y
    
    def joint_2_martinal_type(self, joint_type):    
        marginal = joint_type.sum(axis=0)
        i_T_marginal = composition_to_index(marginal)
        return i_T_marginal, marginal


    def joint_2_ix(self, joint):
        return composition_to_index(joint.flatten())

    def iterate_cond(self):
        c = 0
        for nn in self.len_cond_given_prime:
            yield c, c+nn
            c += nn
            
    def enumerate(self):
        for i_T_y, T_y in enumerate(enumerate_type_class(self.n, self.k_y)):
            for i_T_x_given_y, T_yx in enumerate(enumerate_conditional_type_class(T_y, self.k_x)):
                                
                yield i_T_y, T_y, i_T_x_given_y, T_yx, log_size_conditional_type_class(T_yx)
                
            assert i_T_x_given_y + 1 == conditional_composition_count(T_y, self.k_x)

        assert i_T_y+1 == composition_count(self.n, self.k_y)    









def test_joint_source_channel_types(n, k_v, k_x, k_y, extra_tests = False):
        
    cond_vx = conditional_enum(n, k_v, k_x)

    if extra_tests:
        ix2_T_v = -np.ones(cond_vx.len, dtype=np.int32)
        ix2_T_x_given_v = -np.ones(cond_vx.len, dtype=np.int32)
        arr_log_cond_size_x_given_v = -np.ones(cond_vx.len)

    arr_P_x_given_v = np.random.random(cond_vx.len)
    for st,ed in cond_vx.iterate_cond():
        arr_P_x_given_v[st:ed] = arr_P_x_given_v[st:ed]/arr_P_x_given_v[st:ed].sum()


    ix2_T_v_x_given_v = -np.ones(cond_vx.len, dtype=np.int32)
    for i_T_v, T_v, i_T_x_given_v, T_vx,log_size in cond_vx.enumerate():        
        i_T_vx = cond_vx.joint_2_ix(T_vx)
        
        if extra_tests:
            arr_log_cond_size_x_given_v[i_T_vx] = log_size 
            ix2_T_v[i_T_vx] = i_T_v
            ix2_T_x_given_v[i_T_vx] = i_T_x_given_v
            
        ix2_T_v_x_given_v[i_T_vx] = cond_vx.tuple_2_ix(i_T_v, i_T_x_given_v)
        
    if extra_tests:                
        assert round(np.exp(arr_log_cond_size_x_given_v + cond_vx.size_cond_given_prime[ix2_T_v]).sum()) == (k_x*k_v)**n
    
    cond_y_vx = conditional_enum(n, k_y, k_v*k_x)
    
    if extra_tests:                
        arr_log_cond_size_vx_given_y = -np.ones(cond_y_vx.len)
        ixy2_T_v = -np.ones(cond_y_vx.len, dtype=np.int32)
        ixy2_T_x_given_v = -np.ones(cond_y_vx.len, dtype=np.int32)    
        
    ixy2_log_ratio = np.zeros(cond_y_vx.len)
    ixy_2_T_vx = -np.ones(cond_y_vx.len, dtype=np.int32)
    
    for ix,(i_T_y, T_y, i_T_vx_given_y, T_yvx,log_size) in enumerate(cond_y_vx.enumerate()):
    
        i_T_vx, T_vx = cond_y_vx.joint_2_martinal_type(T_yvx)
        T_vx = T_vx.reshape(k_v, k_x)
            
        ixy2_log_ratio[ix] = log_size-log_size_conditional_type_class(T_vx) # arr_log_cond_size_x_given_v[i_T_vx]
        ixy_2_T_vx[ix] = i_T_vx
        
                    
        if extra_tests:
            arr_log_cond_size_vx_given_y[ix] = log_size
            ixy2_T_v[ix] = ix2_T_v[i_T_vx]
            ixy2_T_x_given_v[ix] = ix2_T_x_given_v[i_T_vx]
            T_v = T_vx.sum(axis=1)
            i_T_v = composition_to_index(T_v)
            i_T_x_given_v = matrix_to_index(T_vx)        
                            
            assert ix2_T_v[i_T_vx] == i_T_v
            assert ix2_T_x_given_v[i_T_vx] == i_T_x_given_v
            assert ixy2_log_ratio[ix] == log_size-log_size_conditional_type_class(T_vx)
                
    
    ixy_2_T_x_given_y = ix2_T_v_x_given_v[ixy_2_T_vx]
    
    if extra_tests:    
        assert (ix2_T_v[ixy_2_T_vx] == ixy2_T_v).all()
        assert (ix2_T_x_given_v[ixy_2_T_vx] == ixy2_T_x_given_v).all()
        assert (ixy_2_T_x_given_y == cond_vx.tuple_2_ix(ixy2_T_v, ixy2_T_x_given_v)).all()
        for st,ed in cond_y_vx.iterate_cond():
            assert round(np.exp(arr_log_cond_size_vx_given_y[st:ed]).sum()) == (k_x*k_v)**n
    
    
    tt = arr_P_x_given_v[ixy_2_T_x_given_y]*np.exp(ixy2_log_ratio)
    for st,ed in cond_y_vx.iterate_cond():
        assert round(tt[st:ed].sum()) == k_v**n       

        

if __name__ == "__main__" and False:

    n = 4
    k_v = 4
    k_x = 3 
    k_y = 2
    print(f"Test conditional type enumaration 2: n={n}; k_y={k_y}; k_x={k_x}; k_v={k_v}")
    
    for n in range(2,6):        
        test_joint_source_channel_types(n, k_v, k_x, k_y, extra_tests=True)
        
    n = 4
    k_v = 2
    k_x = 3 
    k_y = 4
    print(f"Test conditional type enumaration 2: n={n}; k_y={k_y}; k_x={k_x}; k_v={k_v}")
    
    for n in range(2,6):        
        test_joint_source_channel_types(n, k_v, k_x, k_y, extra_tests=True)
    




def _enumerate_compositions_numpy(n: int, k: int) -> np.ndarray:
    """
    Return all compositions of n into k non-negative parts as a 2-D int32
    array of shape (C(n+k-1, k-1), k), in the same order as
    enumerate_type_class(n, k).
    """
    if k == 1:
        return np.array([[n]], dtype=np.int32)
    rows = []
    for first in range(n + 1):
        rest = _enumerate_compositions_numpy(n - first, k - 1)
        block = np.empty((len(rest), k), dtype=np.int32)
        block[:, 0]  = first
        block[:, 1:] = rest
        rows.append(block)
    return np.concatenate(rows, axis=0)


def get_joint_source_channel_types(n, k_v, k_x, k_y, log_W, log_P_V):

    cond_vx = conditional_enum(n, k_v, k_x)

    arr_P_x_given_v = np.random.random(cond_vx.len)
    for st, ed in cond_vx.iterate_cond():
        arr_P_x_given_v[st:ed] /= arr_P_x_given_v[st:ed].sum()

    # Small loop over cond_vx — fast (size C(n+k_v*k_x-1, k_v*k_x-1)).
    # Build lookup: flat composition index of T_vx → tuple index in cond_vx.
    max_comp_idx = composition_count(n, k_v * k_x)
    ix2_T_v_x_given_v = -np.ones(max_comp_idx, dtype=np.int32)

    for i_T_v, T_v, i_T_x_given_v, T_vx, _ in cond_vx.enumerate():
        comp_idx = int(cond_vx.joint_2_ix(T_vx))
        ix2_T_v_x_given_v[comp_idx] = cond_vx.tuple_2_ix(i_T_v, i_T_x_given_v)

    cond_y_vx = conditional_enum(n, k_y, k_v * k_x)
    N   = cond_y_vx.len
    k_vx = k_v * k_x

    # ── Vectorized enumeration ────────────────────────────────────────────────
    # For each T_y (a k_y-composition of n) enumerate all conditional rows
    # in numpy, avoiding the slow Python generator.
    #
    # cond_y_vx orders entries: for each T_y, then for each T_{VX|Y} in the
    # order of enumerate_conditional_type_class(T_y, k_vx).
    # enumerate_conditional_type_class builds the cartesian product of
    # enumerate_type_class(T_y[i], k_vx) for each active row i — same order
    # as _enumerate_compositions_numpy.

    all_T_yvx    = np.empty((N, k_y * k_vx), dtype=np.int32)
    all_log_size = np.empty(N)
    all_T_y      = np.empty((N, k_y),        dtype=np.int32)

    # Precompute all k_vx-compositions for every possible row-sum 0..n.
    row_comps = [_enumerate_compositions_numpy(s, k_vx) for s in range(n + 1)]

    pos = 0
    for T_y in enumerate_type_class(n, k_y):
        # Cartesian product of conditional rows for each y.
        # Start with the first active row, then expand.
        block = row_comps[T_y[0]]          # (R0, k_vx)
        for yi in range(1, k_y):
            rows_yi = row_comps[T_y[yi]]   # (Ri, k_vx)
            R0, Ri  = len(block), len(rows_yi)
            new_block = np.empty((R0 * Ri, (yi + 1) * k_vx), dtype=np.int32)
            new_block[:, :yi * k_vx] = np.repeat(block,    Ri, axis=0)
            new_block[:, yi * k_vx:] = np.tile(rows_yi, (R0, 1))
            block = new_block

        B = len(block)   # number of conditional types for this T_y

        all_T_yvx[pos:pos + B] = block
        all_T_y  [pos:pos + B] = T_y[np.newaxis, :]

        # log_size_conditional_type_class for each row in the block.
        # = sum_yi [ gammaln(T_y[yi]+1) - gammaln(block[:, yi*k_vx:(yi+1)*k_vx]+1).sum(1) ]
        block_3d = block.reshape(B, k_y, k_vx)
        all_log_size[pos:pos + B] = (
            gammaln(T_y[np.newaxis, :] + 1)
            - gammaln(block_3d + 1).sum(axis=2)
        ).sum(axis=1)

        pos += B

    # ── Pass 2: vectorized numpy math ────────────────────────────────────────
    T_yvx_4d = all_T_yvx.reshape(N, k_y, k_v, k_x)        # (N, k_y, k_v, k_x)
    T_vx_2d  = T_yvx_4d.sum(axis=1).reshape(N, k_v, k_x)  # (N, k_v, k_x)
    T_v_vx   = T_vx_2d.sum(axis=2)                         # (N, k_v)

    log_size_cond_vx = (
        gammaln(T_v_vx + 1) - gammaln(T_vx_2d + 1).sum(axis=2)
    ).sum(axis=1)

    ixy2_log_ratio = all_log_size - log_size_cond_vx

    T_xy_NkxKy = T_yvx_4d.sum(axis=2).transpose(0, 2, 1)   # (N, k_x, k_y)
    T_v_N      = T_yvx_4d.sum(axis=(1, 3))                  # (N, k_v)
    log_size_Ty = gammaln(n + 1) - gammaln(all_T_y + 1).sum(axis=1)

    ixy_2_log_alpha = (
        (T_xy_NkxKy * log_W[np.newaxis]).sum(axis=(1, 2))
        + (T_v_N * log_P_V[np.newaxis]).sum(axis=1)
        + log_size_Ty
    )

    T_vx_flat         = T_vx_2d.reshape(N, k_v * k_x)
    comp_indices      = batch_composition_to_index(T_vx_flat)
    ixy_2_T_x_given_y = ix2_T_v_x_given_v[comp_indices]

    tt = arr_P_x_given_v[ixy_2_T_x_given_y] * np.exp(ixy2_log_ratio)
    for st, ed in cond_y_vx.iterate_cond():
        assert round(tt[st:ed].sum()) == k_v ** n

    return ixy_2_log_alpha, ixy2_log_ratio, ixy_2_T_x_given_y, cond_y_vx, cond_vx

        