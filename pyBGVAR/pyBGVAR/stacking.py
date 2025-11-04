"""
GVAR stacking module

This module stacks country-specific VAR models into a global GVAR model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.linalg import solve, inv, eigvals
import warnings

from . import utils
from . import helpers


def stack_gvar(xglobal: pd.DataFrame,
               plag: int,
               globalpost: Dict,
               draws: int,
               thin: int,
               trend: bool,
               eigen: bool = True,
               trim: Optional[float] = None,
               verbose: bool = True) -> Dict:
    """
    Stack country models into global GVAR model.
    
    Parameters
    ----------
    xglobal : DataFrame
        Global data matrix.
    plag : int
        Maximum number of lags.
    globalpost : dict
        Dictionary of country model results from BVAR estimation.
    draws : int
        Number of MCMC draws.
    thin : int
        Thinning interval.
    trend : bool
        Whether trend is included.
    eigen : bool
        Whether to check eigenvalue stability.
    trim : float, optional
        Maximum eigenvalue threshold for trimming.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    dict
        Dictionary containing:
        - S_large: Variance-covariance matrices
        - F_large: Coefficient matrices
        - Ginv_large: G inverse matrices
        - A_large: Stacked coefficient matrices
        - F.eigen: Eigenvalues
        - trim.info: Trimming information
    """
    bigT = xglobal.shape[0]
    bigK = xglobal.shape[1]
    cN = list(globalpost.keys())
    thindraws = draws // thin
    
    if trim is None and eigen:
        trim = 1.05
    
    # Initialize storage arrays
    F_large = np.zeros((bigK, bigK, plag, thindraws))
    A_large = np.zeros((bigK, bigK * plag + 1 + (1 if trend else 0), thindraws))
    S_large = np.zeros((bigK, bigK, thindraws))
    Ginv_large = np.zeros((bigK, bigK, thindraws))
    F_eigen = np.zeros(thindraws)
    
    trim_info = "No trimming"
    
    if verbose:
        print(f"Processing {thindraws} draws...")
    
    # Process each draw
    for irep in range(thindraws):
        if verbose and (irep + 1) % 100 == 0:
            print(f"Processing draw {irep + 1}/{thindraws}")
        
        # Initialize matrices for this draw
        a0 = None
        a1 = None
        G = None
        S_post = []
        
        # Collect country model results
        H_matrices = {}
        for pp in range(plag):
            H_matrices[pp] = None
        
        for cc in cN:
            VAR = globalpost[cc]
            W = VAR.get('W')
            
            if W is None:
                continue
            
            M = VAR['Y'].shape[1]
            
            # Extract coefficients from A_store
            if 'A_store' in VAR:
                A_store_cc = VAR['A_store']
                if A_store_cc.shape[2] > irep:
                    A_draw_cc = A_store_cc[:, :, irep]
                else:
                    A_draw_cc = np.median(A_store_cc, axis=2)
            else:
                # Fallback: use identity if no store available
                A_draw_cc = np.eye(M)
                print(f"No A_store found for country {cc}")
                print(VAR)
                print(A_store_cc.shape)
                print(irep)
                print(A_draw_cc.shape)
                print(A_draw_cc)
                print(A_draw_cc.shape)
            
            # Extract Lambda0 (contemporaneous weakly exogenous)
            # This would need proper extraction from A_store based on variable names
            # Simplified: assume structure
            n_regressors = A_draw_cc.shape[0]
            n_vars = A_draw_cc.shape[1]
            
            # Extract endogenous lag coefficients
            plag_cc = min(plag, n_regressors // n_vars)
            Phi_coeffs = []
            for pp in range(1, plag_cc + 1):
                start_idx = n_vars * (pp - 1)
                end_idx = n_vars * pp
                if end_idx <= n_regressors:
                    Phi_coeffs.append(A_draw_cc[start_idx:end_idx, :].T)
                else:
                    Phi_coeffs.append(np.zeros((n_vars, n_vars)))
            print(f"Phi_coeffs: {Phi_coeffs}")


            # Extract Lambda0 and Lambda lags
            # Simplified: assume weakly exogenous start after endogenous
            wex_start = n_vars * plag_cc
            if wex_start < n_regressors:
                Lambda0 = A_draw_cc[wex_start:wex_start + (W.shape[1] - n_vars), :].T
            else:
                Lambda0 = np.zeros((n_vars, W.shape[1] - n_vars))
           
            
            # Construct A matrix: [I, -Lambda0']
            A = np.hstack([np.eye(n_vars), -Lambda0.T])
            
            # Construct B matrices
            for pp in range(plag):
                if pp < len(Phi_coeffs):
                    Phi_pp = Phi_coeffs[pp]
                    # Extract Lambda lag if available
                    lambda_start = wex_start + (W.shape[1] - n_vars)
                    Lambda_pp = np.zeros((n_vars, W.shape[1] - n_vars))
                    if lambda_start + (W.shape[1] - n_vars) * (pp + 1) <= n_regressors:
                        Lambda_pp = A_draw_cc[lambda_start + (W.shape[1] - n_vars) * pp:
                                            lambda_start + (W.shape[1] - n_vars) * (pp + 1), :].T
                    
                    B_pp = np.hstack([Phi_pp, Lambda_pp])
                else:
                    B_pp = np.zeros((n_vars, W.shape[1]))
                
                # Construct H matrix: B * W
                H_pp = B_pp @ W
                
                if H_matrices[pp] is None:
                    H_matrices[pp] = H_pp
                else:
                    H_matrices[pp] = np.vstack([H_matrices[pp], H_pp])
            
            # Stack G matrix
            if G is None:
                G = A @ W
            else:
                G = np.vstack([G, A @ W])
            
            # Extract constant term from A_store
            if 'A_store' in VAR and VAR['A_store'].shape[0] > 0:
                # Find constant term (usually last row or identified by name)
                # Simplified: take last row as constant
                a0_country = VAR['A_store'][-1, :, irep].reshape(-1, 1) if VAR['A_store'].shape[2] > irep else np.median(VAR['A_store'][-1, :, :], axis=1).reshape(-1, 1)
            else:
                a0_country = np.zeros((M, 1))
            
            if a0 is None:
                a0 = a0_country
            else:
                a0 = np.vstack([a0, a0_country])
            
            # Stack trend term if needed
            if trend:
                # Trend would be second to last if constant is last
                if 'A_store' in VAR and VAR['A_store'].shape[0] > 1:
                    a1_country = VAR['A_store'][-2, :, irep].reshape(-1, 1) if VAR['A_store'].shape[2] > irep else np.median(VAR['A_store'][-2, :, :], axis=1).reshape(-1, 1)
                else:
                    a1_country = np.zeros((M, 1))
                
                if a1 is None:
                    a1 = a1_country
                else:
                    a1 = np.vstack([a1, a1_country])
            
            # Collect variance-covariance matrices
            if 'L_store' in VAR and 'Sv_store' in VAR:
                L_post = VAR['L_store'][:, :, irep] if VAR['L_store'].shape[2] > irep else np.median(VAR['L_store'], axis=2)
                Sv_post = VAR['Sv_store'][:, :, irep] if VAR['Sv_store'].shape[2] > irep else np.median(VAR['Sv_store'], axis=2)
                # Construct SIGMA for median time point
                t_med = VAR['Sv_store'].shape[0] // 2
                if M > 1:
                    S_country = L_post @ np.diag(np.exp(Sv_post[t_med, :])) @ L_post.T
                else:
                    S_country = np.array([[np.exp(Sv_post[t_med, 0])]])
            else:
                S_country = np.eye(M)
            
            S_post.append(S_country)
        
        # Compute G inverse
        # Note: solve() requires two arguments (a, b), so use inv() for matrix inverse
        try:
            G_inv = inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
            warnings.warn(f"Singular G matrix at draw {irep + 1}, using pseudo-inverse")
        
        # Stack variance-covariance matrix (block diagonal)
        S_large_draw = _block_diag(S_post)
        S_large[:, :, irep] = S_large_draw
        
        # Compute global constant term
        b0 = G_inv @ a0
        if trend:
            b1 = G_inv @ a1
        else:
            b1 = None
        
        # Compute global lag coefficients
        ALPHA = None
        for kk in range(plag):
            if kk in H_matrices and H_matrices[kk] is not None:
                F_kk = G_inv @ H_matrices[kk]
            else:
                F_kk = np.zeros((bigK, bigK))
            F_large[:, :, kk, irep] = F_kk
            
            if ALPHA is None:
                ALPHA = F_kk
            else:
                ALPHA = np.hstack([ALPHA, F_kk])
        
        # Add constant and trend
        if trend:
            ALPHA = np.hstack([ALPHA, b0, b1])
        else:
            ALPHA = np.hstack([ALPHA, b0])
        
        A_large[:, :, irep] = ALPHA
        Ginv_large[:, :, irep] = G_inv
        
        # Compute eigenvalues for stability check
        if eigen:
            varndxv = [bigK, (1 + (1 if trend else 0)), plag]
            companion_result = helpers.get_companion(ALPHA, varndxv)
            MM = companion_result['MM']
            
            # Extract relevant block for eigenvalues
            MM_var = MM[:bigK * plag, :bigK * plag]
            eigenvals = eigvals(MM_var)
            F_eigen[irep] = np.max(np.abs(eigenvals))
    
    # Trim unstable draws if requested
    if eigen and trim is not None:
        idx = np.where(F_eigen < trim)[0]
        
        if len(idx) < 10:
            raise ValueError("Less than 10 stable draws found. Please re-estimate the model.")
        
        F_large = F_large[:, :, :, idx]
        S_large = S_large[:, :, idx]
        Ginv_large = Ginv_large[:, :, idx]
        A_large = A_large[:, :, idx]
        F_eigen = F_eigen[idx]
        
        trim_info = (f"Trimming leads to {len(idx)} ({len(idx)/thindraws*100:.2f}%) "
                    f"stable draws out of {thindraws} total draws.")
    
    # Set dimension names
    var_names = list(xglobal.columns)
    
    results = {
        'S_large': S_large,
        'F_large': F_large,
        'Ginv_large': Ginv_large,
        'A_large': A_large,
        'F.eigen': F_eigen,
        'trim.info': trim_info
    }
    
    return results


def _block_diag(blocks: List[np.ndarray]) -> np.ndarray:
    """
    Create block diagonal matrix from list of matrices.
    
    Parameters
    ----------
    blocks : list
        List of matrices to place on diagonal.
        
    Returns
    -------
    array
        Block diagonal matrix.
    """
    if not blocks:
        return np.array([])
    
    sizes = [b.shape[0] for b in blocks]
    total_size = sum(sizes)
    result = np.zeros((total_size, total_size))
    
    row_offset = 0
    col_offset = 0
    for block in blocks:
        n, m = block.shape
        result[row_offset:row_offset+n, col_offset:col_offset+m] = block
        row_offset += n
        col_offset += m
    
    return result


def compute_eigenvalues(ALPHA: np.ndarray,
                       varndxv: List[int]) -> float:
    """
    Compute maximum eigenvalue of companion matrix.
    
    Parameters
    ----------
    ALPHA : array
        Coefficient matrix.
    varndxv : list
        [number of variables, number of deterministics, number of lags]
        
    Returns
    -------
    float
        Maximum absolute eigenvalue.
    """
    companion_result = helpers.get_companion(ALPHA, varndxv)
    MM = companion_result['MM']
    
    nn = varndxv[0]
    nl = varndxv[2]
    
    # Extract relevant block
    MM_var = MM[:nn * nl, :nn * nl]
    eigenvals = np.linalg.eigvals(MM_var)
    
    return np.max(np.abs(eigenvals))

