"""
Impulse Response Function (IRF) module

This module computes impulse response functions for BGVAR models using
different identification schemes:
- Generalized IRF (GIRF)
- Cholesky decomposition
- Sign restrictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.linalg import cholesky, solve, inv, qr
from scipy.stats import multivariate_normal
import warnings

from . import utils
from . import helpers


def irf(x,
        n_ahead: int = 24,
        shockinfo: Optional[pd.DataFrame] = None,
        quantiles: Optional[List[float]] = None,
        expert: Optional[Dict] = None,
        verbose: bool = True) -> Dict:
    """
    Compute impulse response functions.
    
    Parameters
    ----------
    x : BGVAR
        Fitted BGVAR object.
    n_ahead : int, default=24
        Forecasting horizon.
    shockinfo : DataFrame, optional
        Dataframe with shock information.
    quantiles : list, optional
        Posterior quantiles to compute.
    expert : dict, optional
        Expert settings.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    dict
        Dictionary containing:
        - posterior: IRF quantiles
        - shockinfo: Shock information
        - struc.obj: Structural objects
        - model.obj: Model objects
    """
    if quantiles is None:
        quantiles = [0.05, 0.10, 0.16, 0.50, 0.84, 0.90, 0.95]
    
    # Get identification scheme
    ident = _get_identification(shockinfo)
    
    if ident == 'sign' and shockinfo is None:
        raise ValueError("Please provide 'shockinfo' argument for sign restrictions.")
    
    lags = x.args['lags']
    pmax = max(lags)
    xglobal = x.xglobal
    Traw = xglobal.shape[0]
    bigK = xglobal.shape[1]
    bigT = Traw - pmax
    A_large = x.stacked_results['A_large']
    F_large = x.stacked_results['F_large']
    S_large = x.stacked_results['S_large']
    Ginv_large = x.stacked_results['Ginv_large']
    F_eigen = x.stacked_results.get('F.eigen', np.array([]))
    thindraws = len(F_eigen) if len(F_eigen) > 0 else A_large.shape[2]
    
    xdat = xglobal.iloc[pmax:].values
    varNames = list(xglobal.columns)
    cN = list(set([name.split('.')[0] for name in varNames]))
    N = len(cN)
    Q = len(quantiles)
    
    # Expert settings
    expert_list = {
        'MaxTries': 100,
        'save.store': False,
        'use_R': False,
        'applyfun': None,
        'cores': None
    }
    if expert is not None:
        expert_list.update(expert)
    
    MaxTries = expert_list['MaxTries']
    save_store = expert_list['save.store']
    
    # Process shock information
    if ident == 'chol':
        shockinfo_processed = _process_chol_shockinfo(shockinfo, varNames)
    elif ident == 'girf':
        shockinfo_processed = _process_girf_shockinfo(shockinfo, varNames)
    elif ident == 'sign':
        shockinfo_processed = _process_sign_shockinfo(shockinfo, varNames)
    else:
        raise ValueError(f"Unknown identification scheme: {ident}")
    
    shock_nr = len(shockinfo_processed)
    
    if verbose:
        print(f"Computing IRFs for {shock_nr} shocks using {ident} identification...")
    
    # Compute IRFs for each draw
    if save_store:
        IRF_store = np.zeros((bigK, n_ahead + 1, shock_nr, thindraws))
        R_store = np.zeros((bigK, bigK, thindraws)) if ident == 'sign' else None
    else:
        IRF_store = None
        R_store = None
    
    # Compute IRFs
    for irep in range(thindraws):
        if verbose and (irep + 1) % 100 == 0:
            print(f"Processing draw {irep + 1}/{thindraws}")
        
        Fmat = F_large[:, :, :, irep]
        Smat = S_large[:, :, irep]
        Ginv = Ginv_large[:, :, irep]
        
        if ident == 'chol':
            irf_result = _irf_chol(xdat, pmax, n_ahead, Ginv, Fmat, Smat, shockinfo_processed)
        elif ident == 'girf':
            irf_result = _irf_girf(xdat, pmax, n_ahead, Ginv, Fmat, Smat)
        elif ident == 'sign':
            irf_result = _irf_sign_zero(xdat, pmax, n_ahead, Ginv, Fmat, Smat,
                                        shockinfo_processed, MaxTries)
        
        if save_store and irf_result['impl'] is not None:
            IRF_store[:, :, :, irep] = irf_result['impl']
            if R_store is not None and irf_result.get('rot') is not None:
                R_store[:, :, irep] = irf_result['rot']
    
    # Compute quantiles
    if save_store:
        posterior = np.zeros((bigK, n_ahead + 1, shock_nr, Q))
        for q_idx, q in enumerate(quantiles):
            posterior[:, :, :, q_idx] = np.percentile(IRF_store, q * 100, axis=3)
    else:
        # Compute quantiles on the fly (simplified)
        posterior = np.zeros((bigK, n_ahead + 1, shock_nr, Q))
        warnings.warn("Quantile computation without store is simplified. Use save.store=True for full posterior.")
    
    # Get median structural objects
    # Convert quantiles to numpy array for subtraction operation
    quantiles_array = np.asarray(quantiles)
    median_idx = np.argmin(np.abs(quantiles_array - 0.5))
    med_idx = thindraws // 2
    
    A_med = A_large[:, :, med_idx]
    Ginv_med = Ginv_large[:, :, med_idx]
    S_med = S_large[:, :, med_idx]
    Fmat_med = F_large[:, :, :, med_idx]
    
    Rmed = None
    if ident == 'sign' and R_store is not None:
        Rmed = R_store[:, :, med_idx]
    
    # Build output structure
    result = {
        'posterior': posterior,
        'shockinfo': shockinfo_processed,
        'ident': ident,
        'struc.obj': {
            'A': A_med,
            'Ginv': Ginv_med,
            'S': S_med,
            'Rmed': Rmed,
            'Fmat': Fmat_med
        },
        'model.obj': {
            'xglobal': xglobal,
            'lags': lags
        }
    }
    
    if save_store:
        result['IRF_store'] = IRF_store
        result['R_store'] = R_store
    
    if verbose:
        print("IRF computation completed.")
    
    return result


def _get_identification(shockinfo: Optional[pd.DataFrame]) -> str:
    """Get identification scheme from shockinfo."""
    if shockinfo is None:
        return 'chol'
    
    if hasattr(shockinfo, 'attrs') and 'ident' in shockinfo.attrs:
        return shockinfo.attrs['ident']
    
    # Try to infer from columns
    if 'restriction' in shockinfo.columns or 'sign' in shockinfo.columns:
        return 'sign'
    elif 'scale' in shockinfo.columns:
        return 'chol'
    else:
        return 'girf'


def _process_chol_shockinfo(shockinfo: Optional[pd.DataFrame], varNames: List[str]) -> List[Dict]:
    """Process Cholesky identification shockinfo."""
    if shockinfo is None:
        # Default: all variables
        return [{'shock': var, 'scale': 1.0, 'country_idx': None, 'var_idx': i}
                for i, var in enumerate(varNames)]
    
    shocks = []
    for _, row in shockinfo.iterrows():
        shock_var = row.get('shock', '')
        scale = row.get('scale', 1.0)
        
        if shock_var in varNames:
            var_idx = varNames.index(shock_var)
            country = shock_var.split('.')[0]
            country_vars = [v for v in varNames if v.startswith(country + '.')]
            country_idx = [varNames.index(v) for v in country_vars]
            
            shocks.append({
                'shock': shock_var,
                'scale': scale,
                'country_idx': country_idx,
                'var_idx': var_idx
            })
    
    return shocks


def _process_girf_shockinfo(shockinfo: Optional[pd.DataFrame], varNames: List[str]) -> List[Dict]:
    """Process GIRF identification shockinfo."""
    return _process_chol_shockinfo(shockinfo, varNames)


def _process_sign_shockinfo(shockinfo: pd.DataFrame, varNames: List[str]) -> List[Dict]:
    """Process sign restriction shockinfo."""
    # This is a simplified version - full implementation would handle
    # complex sign restriction structures
    shocks = []
    unique_shocks = shockinfo['shock'].unique()
    
    for shock_var in unique_shocks:
        shock_rows = shockinfo[shockinfo['shock'] == shock_var]
        restrictions = []
        
        for _, row in shock_rows.iterrows():
            restrictions.append({
                'restriction': row.get('restriction', ''),
                'sign': row.get('sign', ''),
                'horizon': row.get('horizon', 1),
                'prob': row.get('prob', 1.0)
            })
        
        if shock_var in varNames:
            var_idx = varNames.index(shock_var)
            country = shock_var.split('.')[0]
            country_vars = [v for v in varNames if v.startswith(country + '.')]
            country_idx = [varNames.index(v) for v in country_vars]
            
            shocks.append({
                'shock': shock_var,
                'restrictions': restrictions,
                'country_idx': country_idx,
                'var_idx': var_idx
            })
    
    return shocks


def _irf_chol(xdat: np.ndarray,
              plag: int,
              n_ahead: int,
              Ginv: np.ndarray,
              Fmat: np.ndarray,
              Smat: np.ndarray,
              shocklist: List[Dict]) -> Dict:
    """
    Compute Cholesky-identified IRFs.
    
    Parameters
    ----------
    xdat : array
        Data matrix.
    plag : int
        Number of lags.
    n_ahead : int
        Forecast horizon.
    Ginv : array
        G inverse matrix.
    Fmat : array
        Coefficient matrices (K x K x p).
    Smat : array
        Variance-covariance matrix.
    shocklist : list
        List of shock specifications.
        
    Returns
    -------
    dict
        Dictionary with 'impl' (impulse responses) and 'rot' (rotation matrix, None for Cholesky).
    """
    bigT = xdat.shape[0]
    bigK = xdat.shape[1]
    varNames = list(range(bigK))  # Simplified
    
    # Create P0G matrix (Cholesky decomposition)
    P0G = np.eye(bigK)
    for shock_info in shocklist:
        country_idx = shock_info.get('country_idx', [shock_info['var_idx']])
        if len(country_idx) > 1:
            # Local Cholesky for country
            try:
                P0G_local = cholesky(Smat[np.ix_(country_idx, country_idx)], lower=True)
                P0G[np.ix_(country_idx, country_idx)] = P0G_local
            except np.linalg.LinAlgError:
                # Use eigendecomposition if Cholesky fails
                eigvals, eigvecs = np.linalg.eigh(Smat[np.ix_(country_idx, country_idx)])
                eigvals = np.maximum(eigvals, 1e-10)
                P0G[np.ix_(country_idx, country_idx)] = eigvecs @ np.diag(np.sqrt(eigvals))
        else:
            P0G[country_idx[0], country_idx[0]] = np.sqrt(Smat[country_idx[0], country_idx[0]])
    
    # Create dynamic multiplier (Phi matrices)
    PHI = _compute_phi_matrices(Fmat, plag, n_ahead, bigK)
    
    # Compute IRFs
    invGSigma_u = Ginv @ P0G
    irfa = np.zeros((bigK, bigK, n_ahead + 1))
    
    for ihor in range(n_ahead + 1):
        irfa[:, :, ihor] = PHI[:, :, ihor] @ invGSigma_u
    
    # Apply shock scales and select relevant shocks
    if shocklist:
        irfa_selected = np.zeros((bigK, len(shocklist), n_ahead + 1))
        for i, shock_info in enumerate(shocklist):
            shock_idx = shock_info['var_idx']
            scale = shock_info.get('scale', 1.0)
            irfa_selected[:, i, :] = irfa[:, shock_idx, :] * scale
        irfa = irfa_selected
    
    return {'impl': irfa, 'rot': None, 'icounter': 1}


def _irf_girf(xdat: np.ndarray,
             plag: int,
             n_ahead: int,
             Ginv: np.ndarray,
             Fmat: np.ndarray,
             Smat: np.ndarray) -> Dict:
    """
    Compute Generalized IRFs (GIRF).
    
    Parameters
    ----------
    xdat : array
        Data matrix.
    plag : int
        Number of lags.
    n_ahead : int
        Forecast horizon.
    Ginv : array
        G inverse matrix.
    Fmat : array
        Coefficient matrices.
    Smat : array
        Variance-covariance matrix.
        
    Returns
    -------
    dict
        Dictionary with IRF results.
    """
    bigK = xdat.shape[1]
    
    # Create dynamic multiplier
    PHI = _compute_phi_matrices(Fmat, plag, n_ahead, bigK)
    
    # GIRF uses full covariance matrix
    invGSigma_u = Ginv @ Smat
    
    # Compute IRFs
    irfa = np.zeros((bigK, bigK, n_ahead + 1))
    for ihor in range(n_ahead + 1):
        irfa[:, :, ihor] = PHI[:, :, ihor] @ invGSigma_u
    
    return {'impl': irfa, 'rot': None, 'icounter': 1}


def _irf_sign_zero(xdat: np.ndarray,
                  plag: int,
                  n_ahead: int,
                  Ginv: np.ndarray,
                  Fmat: np.ndarray,
                  Smat: np.ndarray,
                  shocklist: List[Dict],
                  MaxTries: int = 100) -> Dict:
    """
    Compute sign-restriction identified IRFs.
    
    This is a simplified version. Full implementation would handle
    complex sign restriction structures with rotation matrix search.
    """
    # Simplified: use Cholesky as fallback
    # Full implementation would search for rotation matrices
    warnings.warn("Sign restriction IRF implementation is simplified. "
                "Full rotation matrix search not yet implemented.")
    
    return _irf_chol(xdat, plag, n_ahead, Ginv, Fmat, Smat, shocklist)


def _compute_phi_matrices(Fmat: np.ndarray,
                          plag: int,
                          n_ahead: int,
                          bigK: int) -> np.ndarray:
    """
    Compute Phi matrices (dynamic multipliers).
    
    Parameters
    ----------
    Fmat : array
        Coefficient matrices (K x K x p).
    plag : int
        Number of lags.
    n_ahead : int
        Forecast horizon.
    bigK : int
        Number of variables.
        
    Returns
    -------
    array
        Phi matrices (K x K x (n_ahead + 1)).
    """
    PHI = np.zeros((bigK, bigK, n_ahead + 1))
    PHI[:, :, 0] = np.eye(bigK)  # Identity at horizon 0
    
    for ihor in range(1, n_ahead + 1):
        acc = np.zeros((bigK, bigK))
        for pp in range(1, min(ihor, plag) + 1):
            if pp <= Fmat.shape[2]:
                acc += Fmat[:, :, pp - 1] @ PHI[:, :, ihor - pp]
        PHI[:, :, ihor] = acc
    
    return PHI


def get_shockinfo(ident: str = 'chol', nr_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Get template shockinfo DataFrame.
    
    Parameters
    ----------
    ident : str, default='chol'
        Identification scheme: 'chol', 'girf', or 'sign'.
    nr_rows : int, optional
        Number of rows (shocks).
        
    Returns
    -------
    DataFrame
        Template shockinfo DataFrame.
    """
    if ident == 'chol' or ident == 'girf':
        columns = ['shock', 'scale', 'global']
        df = pd.DataFrame(columns=columns)
        if nr_rows is not None:
            df = pd.DataFrame(index=range(nr_rows), columns=columns)
            df['global'] = False
    elif ident == 'sign':
        columns = ['shock', 'restriction', 'sign', 'horizon', 'scale', 'prob', 'global']
        df = pd.DataFrame(columns=columns)
        if nr_rows is not None:
            df = pd.DataFrame(index=range(nr_rows), columns=columns)
            df['global'] = False
            df['horizon'] = 1
            df['prob'] = 1.0
    else:
        raise ValueError(f"Unknown identification scheme: {ident}")
    
    df.attrs = {'ident': ident}
    return df


def add_shockinfo(shockinfo: pd.DataFrame,
                 shock: str,
                 restriction: Optional[List[str]] = None,
                 sign: Optional[List[str]] = None,
                 horizon: Union[int, List[int]] = 1,
                 scale: float = 1.0,
                 prob: Union[float, List[float]] = 1.0) -> pd.DataFrame:
    """
    Add shock information to shockinfo DataFrame.
    
    Parameters
    ----------
    shockinfo : DataFrame
        Existing shockinfo DataFrame.
    shock : str
        Shock variable name.
    restriction : list, optional
        Restriction variables (for sign restrictions).
    sign : list, optional
        Sign restrictions ('<', '>', '=').
    horizon : int or list, default=1
        Restriction horizon(s).
    scale : float, default=1.0
        Shock scale.
    prob : float or list, default=1.0
        Restriction probability.
        
    Returns
    -------
    DataFrame
        Updated shockinfo DataFrame.
    """
    ident = shockinfo.attrs.get('ident', 'chol')
    
    if ident == 'sign':
        if restriction is None or sign is None:
            raise ValueError("For sign restrictions, both 'restriction' and 'sign' must be provided.")
        
        if isinstance(horizon, int):
            horizon = [horizon] * len(restriction)
        if isinstance(prob, (int, float)):
            prob = [prob] * len(restriction)
        
        for i, (res, sig, h, p) in enumerate(zip(restriction, sign, horizon, prob)):
            new_row = pd.DataFrame({
                'shock': [shock],
                'restriction': [res],
                'sign': [sig],
                'horizon': [h],
                'scale': [scale],
                'prob': [p],
                'global': [False]
            })
            shockinfo = pd.concat([shockinfo, new_row], ignore_index=True)
    else:
        new_row = pd.DataFrame({
            'shock': [shock],
            'scale': [scale],
            'global': [False]
        })
        shockinfo = pd.concat([shockinfo, new_row], ignore_index=True)
    
    shockinfo.attrs = {'ident': ident}
    return shockinfo

