"""
Diagnostic functions for BGVAR models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats
import warnings


def conv_diag(x: 'BGVAR', crit_val: float = 1.96) -> Dict:
    """
    MCMC convergence diagnostics using Geweke test.
    
    Parameters
    ----------
    x : BGVAR
        Fitted BGVAR object.
    crit_val : float, default=1.96
        Critical value for test statistic.
        
    Returns
    -------
    dict
        Dictionary containing Geweke statistics and percentage exceeding threshold.
    """
    ALPHA = x.stacked_results.get('A_large', None)
    if ALPHA is None:
        raise ValueError("No posterior draws available for convergence diagnostics.")
    
    draws = ALPHA.shape[2]
    d1, d2 = ALPHA.shape[0], ALPHA.shape[1]
    K = d1 * d2
    
    geweke_z = []
    for i in range(d1):
        for j in range(d2):
            chain = ALPHA[i, j, :]
            if len(chain) > 20:  # Need sufficient draws
                # Simple Geweke test: compare first 10% and last 50%
                n1 = max(int(len(chain) * 0.1), 10)
                n2 = max(int(len(chain) * 0.5), 10)
                
                mean1 = np.mean(chain[:n1])
                mean2 = np.mean(chain[-n2:])
                
                var1 = np.var(chain[:n1], ddof=1)
                var2 = np.var(chain[-n2:], ddof=1)
                
                se = np.sqrt(var1 / n1 + var2 / n2)
                if se > 0:
                    z = (mean1 - mean2) / se
                    geweke_z.append(z)
    
    idx = [i for i, z in enumerate(geweke_z) if abs(z) > crit_val]
    perc = f"{len(idx)} out of {len(geweke_z)} variables' z-values exceed the {crit_val} threshold ({len(idx)/len(geweke_z)*100:.2f}%)."
    
    return {
        'geweke.z': np.array(geweke_z),
        'perc': perc
    }


def resid_corr_test(obj: 'BGVAR', lag_cor: int = 1, alpha: float = 0.95,
                   dig1: int = 5, dig2: int = 3) -> Dict:
    """
    F-test for serial autocorrelation in residuals.
    
    Parameters
    ----------
    obj : BGVAR
        Fitted BGVAR object.
    lag_cor : int, default=1
        Order of serial correlation to test.
    alpha : float, default=0.95
        Significance level.
    dig1 : int, default=5
        Digits for F-statistics.
    dig2 : int, default=3
        Digits for p-values.
        
    Returns
    -------
    dict
        Dictionary containing F-statistics and p-values.
    """
    xglobal = obj.xglobal
    res = obj.cc_results.get('res', {})
    lags = obj.args['lags']
    pmax = max(lags)
    bigT = xglobal.shape[0] - pmax
    
    varNames = list(xglobal.columns)
    cN = list(set([name.split('.')[0] for name in varNames]))
    
    Fstat = {}
    pL = {}
    critL = {}
    
    for cc in cN:
        if cc not in res:
            continue
        
        idx = [i for i, name in enumerate(varNames) if name.startswith(cc + '.')]
        X_dat = xglobal.iloc[pmax:, idx].values
        r_dat = res[cc]
        
        ki = X_dat.shape[1]
        dof = bigT - ki - lag_cor
        
        if dof <= 0:
            continue
        
        # Projection matrix
        try:
            M = np.eye(bigT) - X_dat @ np.linalg.solve(X_dat.T @ X_dat, X_dat.T)
        except np.linalg.LinAlgError:
            M = np.eye(bigT) - X_dat @ np.linalg.pinv(X_dat.T @ X_dat) @ X_dat.T
        
        faux = []
        pV = []
        
        for j in range(ki):
            # Construct lagged residuals
            w = np.zeros((bigT, lag_cor))
            for p in range(1, lag_cor + 1):
                w[p:, p-1] = r_dat[:-p, j]
            
            # F-statistic
            num = bigT * (r_dat[:, j].T @ w @ np.linalg.solve(w.T @ M @ w, w.T @ r_dat[:, j])) / (r_dat[:, j].T @ r_dat[:, j])
            F_stat = (dof / lag_cor) * (num / (bigT - num))
            faux.append(F_stat)
            
            # P-value
            p_val = 1 - stats.f.cdf(F_stat, lag_cor, dof)
            pV.append(p_val)
        
        Fstat[cc] = np.array(faux)
        pL[cc] = np.array(pV)
        critL[cc] = stats.f.ppf(alpha, lag_cor, dof)
    
    # Summary table
    pp = np.concatenate(list(pL.values()))
    p_res = np.zeros((4, 2))
    K = len(pp)
    
    p_res[0, :] = [np.sum(pp > 0.10), np.sum(pp > 0.10) / K * 100]
    p_res[1, :] = [np.sum((pp > 0.05) & (pp <= 0.10)), np.sum((pp > 0.05) & (pp <= 0.10)) / K * 100]
    p_res[2, :] = [np.sum((pp > 0.01) & (pp <= 0.05)), np.sum((pp > 0.01) & (pp <= 0.05)) / K * 100]
    p_res[3, :] = [np.sum(pp <= 0.01), np.sum(pp <= 0.01) / K * 100]
    
    return {
        'Fstat': Fstat,
        'pL': pL,
        'p.res': p_res
    }


def avg_pair_cc(object: Union['BGVAR', 'bgvar.resid'],
                digits: int = 3) -> Dict:
    """
    Compute average pairwise cross-sectional correlations.
    
    Parameters
    ----------
    object : BGVAR or bgvar.resid
        BGVAR object or residuals object.
    digits : int, default=3
        Number of digits for output.
        
    Returns
    -------
    dict
        Dictionary containing correlation statistics.
    """
    if hasattr(object, 'xglobal'):
        # BGVAR object
        lags = object.args['lags']
        pmax = max(lags)
        dat = object.xglobal.iloc[pmax:].values
        res = np.concatenate([object.cc_results['res'][cc] for cc in object.cN], axis=1)
        
        # Reorder to match data columns
        varNames = list(object.xglobal.columns)
        res = res[:, [varNames.index(col) for col in varNames if col in varNames]]
    else:
        # Residuals object
        dat = object.get('Data', None)
        res = object.get('country', None)
        if res is not None:
            res = np.mean(res, axis=0)
    
    bigT = res.shape[0]
    varNames = list(dat.columns) if isinstance(dat, pd.DataFrame) else [f'var{i}' for i in range(dat.shape[1])]
    
    cN = list(set([name.split('.')[0] if '.' in name else name[:2] for name in varNames]))
    vars_list = list(set([name.split('.', 1)[1] if '.' in name else name[2:] for name in varNames]))
    
    # Compute correlations by variable
    data_cor = {}
    resid_cor = {}
    
    for var in vars_list:
        var_cols = [i for i, name in enumerate(varNames) if name.endswith('.' + var) or name == var]
        if len(var_cols) > 1:
            data_cor_var = np.corrcoef(dat[:, var_cols].T) if not isinstance(dat, pd.DataFrame) else dat.iloc[:, var_cols].corr().values
            resid_cor_var = np.corrcoef(res[:, var_cols].T)
            
            # Remove diagonal and compute average
            np.fill_diagonal(data_cor_var, np.nan)
            np.fill_diagonal(resid_cor_var, np.nan)
            
            data_cor[var] = np.nanmean(np.abs(data_cor_var))
            resid_cor[var] = np.nanmean(np.abs(resid_cor_var))
    
    # Summary statistics
    data_res = np.zeros((4, len(data_cor)))
    res_res = np.zeros((4, len(resid_cor)))
    
    for i, var in enumerate(data_cor.keys()):
        cor_val = abs(data_cor[var])
        res_val = abs(resid_cor[var])
        
        data_res[:, i] = [
            int(cor_val <= 0.1),
            int((cor_val > 0.1) & (cor_val <= 0.2)),
            int((cor_val > 0.2) & (cor_val <= 0.5)),
            int(cor_val > 0.5)
        ]
        
        res_res[:, i] = [
            int(res_val <= 0.1),
            int((res_val > 0.1) & (res_val <= 0.2)),
            int((res_val > 0.2) & (res_val <= 0.5)),
            int(res_val > 0.5)
        ]
    
    return {
        'data.cor': data_cor,
        'resid.cor': resid_cor,
        'data.res': data_res,
        'res.res': res_res
    }

