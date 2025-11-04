"""
Historical Decomposition (HD) module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.linalg import cholesky, inv
import warnings

from . import utils


def hd(x,
       var_slct: Optional[List[str]] = None,
       verbose: bool = True) -> Dict:
    """
    Compute Historical Decomposition.
    
    Parameters
    ----------
    x : bgvar.irf
        IRF object from irf() function.
    var_slct : list, optional
        Variables to decompose.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    dict
        Dictionary containing historical decomposition.
    """
    if verbose:
        print("Start computing historical decomposition...")
    
    xglobal = x['model.obj']['xglobal']
    lags = x['model.obj']['lags']
    pmax = max(lags)
    ident = x.get('ident', 'chol')
    
    A = x['struc.obj']['A']
    Fmat = x['struc.obj']['Fmat']
    Ginv = x['struc.obj']['Ginv']
    Smat = x['struc.obj']['S']
    Rmed = x['struc.obj'].get('Rmed', None)
    
    varNames = list(xglobal.columns)
    bigK = len(varNames)
    bigT = xglobal.shape[0] - pmax
    
    if var_slct is None:
        var_slct = varNames
    else:
        if not all(v in varNames for v in var_slct):
            raise ValueError("One of the variables is not in the system.")
    
    # Get structural shocks
    if ident == 'sign':
        rotation_matrix = Rmed
        if rotation_matrix is None:
            raise ValueError("No rotation matrix available.")
    else:
        rotation_matrix = np.eye(bigK)
    
    # Compute historical decomposition
    HD_result = _compute_hd(
        xglobal, A, Fmat, Ginv, Smat, rotation_matrix,
        pmax, bigT, bigK, varNames, var_slct
    )
    
    result = {
        'HD': HD_result,
        'xglobal': xglobal,
        'var_slct': var_slct
    }
    
    if verbose:
        print("Historical decomposition completed.")
    
    return result


def _compute_hd(xglobal: pd.DataFrame,
               A: np.ndarray,
               Fmat: np.ndarray,
               Ginv: np.ndarray,
               Smat: np.ndarray,
               rotation_matrix: np.ndarray,
               pmax: int,
               bigT: int,
               bigK: int,
               varNames: List[str],
               var_slct: List[str]) -> np.ndarray:
    """
    Compute historical decomposition.
    
    Parameters
    ----------
    xglobal : DataFrame
        Global data.
    A : array
        Coefficient matrix.
    Fmat : array
        Lag coefficient matrices.
    Ginv : array
        G inverse.
    Smat : array
        Variance-covariance matrix.
    rotation_matrix : array
        Rotation matrix.
    pmax : int
        Maximum lag.
    bigT : int
        Sample size.
    bigK : int
        Number of variables.
    varNames : list
        Variable names.
    var_slct : list
        Selected variables.
        
    Returns
    -------
    array
        Historical decomposition (T x K x K).
    """
    # Get data
    xdat = xglobal.iloc[pmax:].values
    
    # Compute structural shocks
    residuals = _compute_residuals(xdat, A, Fmat, pmax, bigT, bigK)
    Sigma_u = Ginv @ Smat @ Ginv.T
    C = cholesky(Sigma_u, lower=True) @ rotation_matrix
    structural_shocks = residuals @ inv(C.T)
    
    # Compute historical decomposition
    HD = np.zeros((bigT, bigK, bigK))
    plag = Fmat.shape[2]
    
    # Compute Phi matrices
    PHI = _compute_phi_for_hd(Fmat, plag, bigT, bigK)
    
    # Decompose each variable's history
    for t in range(bigT):
        for j in range(bigK):
            contrib = np.zeros(bigK)
            for s in range(min(t + 1, bigT)):
                if t - s < PHI.shape[2]:
                    contrib += PHI[:, j, t - s] * structural_shocks[s, j]
            HD[t, :, j] = contrib
    
    return HD


def _compute_residuals(xdat: np.ndarray,
                      A: np.ndarray,
                      Fmat: np.ndarray,
                      pmax: int,
                      bigT: int,
                      bigK: int) -> np.ndarray:
    """Compute residuals from fitted model."""
    residuals = np.zeros((bigT, bigK))
    plag = Fmat.shape[2]
    
    for t in range(pmax, xdat.shape[0]):
        y_pred = np.zeros(bigK)
        
        # Add lag terms
        for p in range(plag):
            if t - p - 1 >= 0:
                y_pred += Fmat[:, :, p] @ xdat[t - p - 1, :]
        
        # Add constant (last column of A)
        if A.shape[1] > bigK * plag:
            y_pred += A[:, -1]  # Constant term
        
        residuals[t - pmax, :] = xdat[t, :] - y_pred
    
    return residuals


def _compute_phi_for_hd(Fmat: np.ndarray,
                        plag: int,
                        bigT: int,
                        bigK: int) -> np.ndarray:
    """Compute Phi matrices for HD."""
    max_horizon = min(bigT, 100)  # Limit horizon for computational efficiency
    PHI = np.zeros((bigK, bigK, max_horizon))
    PHI[:, :, 0] = np.eye(bigK)
    
    for h in range(1, max_horizon):
        acc = np.zeros((bigK, bigK))
        for pp in range(1, min(h + 1, plag + 1)):
            if pp <= Fmat.shape[2]:
                acc += Fmat[:, :, pp - 1] @ PHI[:, :, h - pp]
        PHI[:, :, h] = acc
    
    return PHI

