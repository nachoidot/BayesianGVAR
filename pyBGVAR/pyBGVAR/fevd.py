"""
Forecast Error Variance Decomposition (FEVD) module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.linalg import cholesky
import warnings

from . import utils
from . import irf


def fevd(x,
         rotation_matrix: Optional[np.ndarray] = None,
         var_slct: Optional[List[str]] = None,
         verbose: bool = True) -> Dict:
    """
    Compute Forecast Error Variance Decomposition.
    
    Parameters
    ----------
    x : bgvar.irf
        IRF object from irf() function.
    rotation_matrix : array, optional
        Rotation matrix for sign-identified shocks.
    var_slct : list, optional
        Variables to compute FEVD for.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    dict
        Dictionary containing:
        - FEVD: Forecast error variance decomposition array
        - xglobal: Data used
    """
    if verbose:
        print("Start computing forecast error variance decomposition...")
    
    xglobal = x['model.obj']['xglobal']
    lags = x['model.obj']['lags']
    pmax = max(lags)
    ident = x.get('ident', 'chol')
    Traw = xglobal.shape[0]
    bigK = xglobal.shape[1]
    xdat = xglobal.iloc[pmax:].values
    bigT = xdat.shape[0]
    
    A = x['struc.obj']['A']
    Fmat = x['struc.obj']['Fmat']
    Ginv = x['struc.obj']['Ginv']
    Smat = x['struc.obj']['S']
    Rmed = x['struc.obj'].get('Rmed', None)
    
    horizon = x['posterior'].shape[1]
    varNames = list(xglobal.columns)
    
    if ident == 'sign':
        if rotation_matrix is None:
            rotation_matrix = Rmed
        if rotation_matrix is None or np.any(np.isnan(rotation_matrix)):
            raise ValueError("No rotation matrix available for sign restrictions.")
    else:
        rotation_matrix = np.eye(bigK)
    
    if var_slct is None:
        var_slct = varNames
        if verbose:
            print("FEVD computed for all variables.")
    else:
        if not all(v in varNames for v in var_slct):
            raise ValueError("One of the variables is not in the system.")
        if verbose:
            print(f"FEVD computed for variables: {', '.join(var_slct)}")
    
    # Compute FEVD
    FEVD = _compute_fevd(Fmat, Ginv, Smat, rotation_matrix, horizon, bigK, varNames, var_slct)
    
    result = {
        'FEVD': FEVD,
        'xglobal': xglobal
    }
    
    if verbose:
        print("FEVD computation completed.")
    
    return result


def _compute_fevd(Fmat: np.ndarray,
                 Ginv: np.ndarray,
                 Smat: np.ndarray,
                 rotation_matrix: np.ndarray,
                 horizon: int,
                 bigK: int,
                 varNames: List[str],
                 var_slct: List[str]) -> np.ndarray:
    """
    Compute FEVD from structural model.
    
    Parameters
    ----------
    Fmat : array
        Coefficient matrices (K x K x p).
    Ginv : array
        G inverse matrix.
    Smat : array
        Variance-covariance matrix.
    rotation_matrix : array
        Rotation matrix.
    horizon : int
        Forecast horizon.
    bigK : int
        Number of variables.
    varNames : list
        Variable names.
    var_slct : list
        Selected variables.
        
    Returns
    -------
    array
        FEVD array (K x horizon x K).
    """
    # Compute structural shock matrix
    Sigma_u = Ginv @ Smat @ Ginv.T
    C = cholesky(Sigma_u, lower=True) @ rotation_matrix
    
    # Compute Phi matrices (dynamic multipliers)
    plag = Fmat.shape[2]
    PHI = _compute_phi_matrices_for_fevd(Fmat, plag, horizon, bigK)
    
    # Initialize FEVD array
    FEVD = np.zeros((bigK, horizon, bigK))
    var_indices = [varNames.index(v) for v in var_slct]
    
    # Compute FEVD
    for h in range(horizon):
        # Cumulative contribution of shocks
        cum_contrib = np.zeros((bigK, bigK))
        for j in range(h + 1):
            if j < PHI.shape[2]:
                contrib = PHI[:, :, j] @ C
                cum_contrib += contrib * contrib.T
        
        # Denominator: total variance
        denom = np.diag(cum_contrib)
        denom[denom == 0] = 1.0  # Avoid division by zero
        
        # FEVD as fraction
        for i, var_idx in enumerate(var_indices):
            for j in range(bigK):
                if denom[var_idx] > 0:
                    FEVD[var_idx, h, j] = cum_contrib[var_idx, j] / denom[var_idx]
    
    return FEVD


def _compute_phi_matrices_for_fevd(Fmat: np.ndarray,
                                   plag: int,
                                   horizon: int,
                                   bigK: int) -> np.ndarray:
    """
    Compute Phi matrices for FEVD computation.
    
    Parameters
    ----------
    Fmat : array
        Coefficient matrices.
    plag : int
        Number of lags.
    horizon : int
        Forecast horizon.
    bigK : int
        Number of variables.
        
    Returns
    -------
    array
        Phi matrices (K x K x horizon).
    """
    PHI = np.zeros((bigK, bigK, horizon))
    PHI[:, :, 0] = np.eye(bigK)
    
    for h in range(1, horizon):
        acc = np.zeros((bigK, bigK))
        for pp in range(1, min(h + 1, plag + 1)):
            if pp <= Fmat.shape[2]:
                acc += Fmat[:, :, pp - 1] @ PHI[:, :, h - pp]
        PHI[:, :, h] = acc
    
    return PHI


def gfevd(x: 'BGVAR',
         n_ahead: int = 24,
         running: bool = True,
         var_slct: Optional[List[str]] = None,
         verbose: bool = True) -> Dict:
    """
    Compute Generalized Forecast Error Variance Decomposition (GFEVD).
    
    This function calculates the Lanne-Nyberg (2016) corrected GFEVD which sums to unity,
    based on generalized impulse response functions.
    
    Parameters
    ----------
    x : BGVAR
        Fitted BGVAR object.
    n_ahead : int, default=24
        Forecast horizon.
    running : bool, default=True
        If True, only running mean over posterior draws is calculated.
        If False, full analysis with bounds is computed (memory intensive).
    var_slct : list, optional
        Variables to compute GFEVD for.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    dict
        Dictionary containing:
        - FEVD: GFEVD array (K x K x n_ahead) or (K x K x n_ahead x 3) with bounds
        - xglobal: Global data
        
    References
    ----------
    Lanne, M. and H. Nyberg (2016). Generalized Forecast Error Variance Decomposition 
    for Linear and Nonlinear Multivariate Models. Oxford Bulletin of Economics and Statistics, 
    Vol. 78(4), pp. 595-603.
    """
    import time
    start_time = time.time()
    
    if verbose:
        print("\nStart computing generalized forecast error variance decomposition of Bayesian Global Vector Autoregression.\n")
    
    # Get model objects
    lags = x.args['lags']
    pmax = max(lags)
    xglobal = x.xglobal
    Traw = xglobal.shape[0]
    bigK = xglobal.shape[1]
    Kbig = pmax * bigK
    bigT = Traw - pmax
    
    A_large = x.stacked_results['A_large']
    F_large = x.stacked_results['F_large']
    S_large = x.stacked_results['S_large']
    Ginv_large = x.stacked_results['Ginv_large']
    F_eigen = x.stacked_results.get('F_eigen', list(range(A_large.shape[2])))
    
    xdata = xglobal.iloc[pmax:].values
    thindraws = len(F_eigen)
    varNames = list(xglobal.columns)
    
    if verbose:
        print(f"Start computation with {thindraws} stable draws in total.")
    
    if running:
        # Running mean only
        GFEVD_post = np.zeros((bigK, bigK, n_ahead))
        thindraws2 = 0
        
        for irep in range(thindraws):
            # Compute GIRF for this draw
            invG = Ginv_large[:, :, irep]
            lF = F_large[:, :, :, irep]
            gcov = S_large[:, :, irep]
            
            # Compute impulse responses using GIRF logic
            irfa = _compute_girf_for_fevd(invG, lF, gcov, xdata, n_ahead)
            
            # Compute GFEVD from impulse responses
            GFEVD_draw = _mk_fevd_sims(irfa)
            
            if not np.any(np.isnan(GFEVD_draw)):
                GFEVD_post += GFEVD_draw
                thindraws2 += 1
        
        if thindraws2 > 0:
            GFEVD_post = GFEVD_post / thindraws2
        else:
            warnings.warn("No valid GFEVD draws computed!")
        
        result = {
            'FEVD': GFEVD_post,
            'xglobal': xglobal,
            'R': None
        }
        
    else:
        # Full calculation with bounds (memory intensive)
        GFEVD_draws = np.zeros((thindraws, bigK, bigK, n_ahead))
        GFEVD_post = np.zeros((bigK, bigK, n_ahead, 3))
        
        for irep in range(thindraws):
            invG = Ginv_large[:, :, irep]
            lF = F_large[:, :, :, irep]
            gcov = S_large[:, :, irep]
            
            irfa = _compute_girf_for_fevd(invG, lF, gcov, xdata, n_ahead)
            GFEVD_draws[irep] = _mk_fevd_sims(irfa)
        
        # Compute quantiles
        GFEVD_post[:, :, :, 0] = np.quantile(GFEVD_draws, 0.16, axis=0)
        GFEVD_post[:, :, :, 1] = np.median(GFEVD_draws, axis=0)
        GFEVD_post[:, :, :, 2] = np.quantile(GFEVD_draws, 0.84, axis=0)
        
        result = {
            'FEVD': GFEVD_post,
            'xglobal': xglobal,
            'R': None,
            'GFEVD_store': GFEVD_draws
        }
    
    elapsed = time.time() - start_time
    if verbose:
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"\nNeeded time for computation: {mins} min {secs} seconds.\n")
    
    return result


def _compute_girf_for_fevd(invG: np.ndarray,
                           lF: np.ndarray,
                           gcov: np.ndarray,
                           x: np.ndarray,
                           horizon: int) -> np.ndarray:
    """
    Helper function to compute GIRFs for FEVD calculation.
    
    Parameters
    ----------
    invG : ndarray
        Inverse of G matrix.
    lF : ndarray
        F matrix (companion form).
    gcov : ndarray
        Covariance matrix.
    x : ndarray
        Data.
    horizon : int
        Forecast horizon.
        
    Returns
    -------
    ndarray
        Impulse response array (K x K x horizon).
    """
    bigK = gcov.shape[0]
    impl = np.zeros((bigK, bigK, horizon))
    
    # Compute standard deviations
    sigma = np.sqrt(np.diag(gcov))
    
    # Compute impulse responses for each variable
    for j in range(bigK):
        # Shock to variable j
        shock_vec = np.zeros(bigK)
        shock_vec[j] = sigma[j]
        
        # Transform shock through inverse G
        eps = invG @ shock_vec
        
        # Compute impulse responses
        resp = eps.copy()
        impl[:, j, 0] = resp
        
        # Iterate forward
        for h in range(1, horizon):
            resp = lF @ resp
            impl[:, j, h] = resp[:bigK]
    
    return impl


def _mk_fevd_sims(irfa: np.ndarray) -> np.ndarray:
    """
    Compute FEVD from impulse response array.
    
    Parameters
    ----------
    irfa : ndarray
        Impulse response array (K x K x horizon).
        
    Returns
    -------
    ndarray
        FEVD array (K x K x horizon).
    """
    bigK = irfa.shape[0]
    horizon = irfa.shape[2]
    
    fevd = np.zeros((bigK, bigK, horizon))
    
    for i in range(bigK):
        for h in range(horizon):
            # Cumulative squared responses up to horizon h
            cumsum_resp = np.zeros(bigK)
            for j in range(bigK):
                cumsum_resp[j] = np.sum(irfa[i, j, :(h+1)]**2)
            
            # Total variance
            total_var = np.sum(cumsum_resp)
            
            # FEVD: contribution of each shock
            if total_var > 0:
                fevd[i, :, h] = cumsum_resp / total_var
            else:
                fevd[i, :, h] = np.nan
    
    return fevd

