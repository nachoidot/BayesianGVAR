"""
Plotting functions for BGVAR objects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Set style
matplotlib.style.use('seaborn-v0_8-whitegrid')


def plot_bgvar(x, **kwargs):
    """
    Plot BGVAR model diagnostics.
    
    Parameters
    ----------
    x : BGVAR
        Fitted BGVAR object.
    **kwargs
        Additional plotting arguments.
    """
    warnings.warn("Plot functionality for BGVAR objects is simplified.")
    print(f"BGVAR Model Summary:")
    print(f"  Prior: {x.args['prior']}")
    print(f"  Countries: {x.N}")
    print(f"  Variables: {x.xglobal.shape[1]}")
    print(f"  Sample size: {x.args['Traw']}")


def plot_irf(x,
             which: Optional[List[str]] = None,
             resp: Optional[List[str]] = None,
             shock: Optional[List[str]] = None,
             **kwargs) -> plt.Figure:
    """
    Plot impulse response functions.
    
    Parameters
    ----------
    x : bgvar.irf
        IRF object from irf() function.
    which : list, optional
        Which quantiles to plot.
    resp : list, optional
        Response variables to plot.
    shock : list, optional
        Shocks to plot.
    **kwargs
        Additional plotting arguments.
        
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    posterior = x.get('posterior', None)
    if posterior is None:
        raise ValueError("No posterior IRFs available for plotting.")
    
    bigK, horizon, shock_nr, Q = posterior.shape
    
    if which is None:
        which = [0.16, 0.50, 0.84]  # Default: 68% credible intervals
    
    varNames = list(x['model.obj']['xglobal'].columns)
    shockinfo = x.get('shockinfo', [])
    
    if resp is None:
        resp = varNames
    if shock is None:
        shock = list(range(shock_nr))
    
    # Create subplots
    n_resp = len(resp)
    n_shock = len(shock)
    fig, axes = plt.subplots(n_resp, n_shock, figsize=(5*n_shock, 4*n_resp))
    
    if n_resp == 1 and n_shock == 1:
        axes = [[axes]]
    elif n_resp == 1:
        axes = [axes]
    elif n_shock == 1:
        axes = [[ax] for ax in axes]
    
    for i, r in enumerate(resp):
        for j, s in enumerate(shock):
            ax = axes[i][j] if isinstance(axes[0], list) else axes[i] if n_shock == 1 else axes[j]
            
            resp_idx = varNames.index(r) if r in varNames else int(r)
            shock_idx = s if isinstance(s, int) else shock.index(s) if s in shock else int(s)
            
            # Extract IRF
            irf_data = posterior[resp_idx, :, shock_idx, :]
            
            # Plot quantiles
            for q_idx, q in enumerate(which):
                q_val = q if isinstance(q, float) else which.index(q)
                if q_val < Q:
                    ax.plot(irf_data[:, int(q_val)], label=f'{q*100:.0f}%' if isinstance(q, float) else str(q))
            
            # Zero line
            ax.axhline(0, color='red', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Horizon')
            ax.set_ylabel('Response')
            ax.set_title(f'{r} | {shockinfo[shock_idx].get("shock", f"Shock {shock_idx}")}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_fevd(x,
             var_slct: Optional[List[str]] = None,
             **kwargs) -> plt.Figure:
    """
    Plot forecast error variance decomposition.
    
    Parameters
    ----------
    x : bgvar.fevd
        FEVD object from fevd() function.
    var_slct : list, optional
        Variables to plot.
    **kwargs
        Additional plotting arguments.
        
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    FEVD = x.get('FEVD', None)
    if FEVD is None:
        raise ValueError("No FEVD data available for plotting.")
    
    varNames = list(x['xglobal'].columns)
    
    if var_slct is None:
        var_slct = varNames
    
    bigK, horizon, _ = FEVD.shape
    
    # Create subplots
    n_vars = len(var_slct)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 4*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(var_slct):
        var_idx = varNames.index(var)
        ax = axes[i]
        
        # Stacked area plot
        fevd_data = FEVD[var_idx, :, :].T  # (K x horizon)
        
        ax.stackplot(range(horizon), *fevd_data, labels=varNames, alpha=0.7)
        ax.set_xlabel('Horizon')
        ax.set_ylabel('FEVD')
        ax.set_title(f'Forecast Error Variance Decomposition: {var}')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_hd(x,
           var_slct: Optional[List[str]] = None,
           **kwargs) -> plt.Figure:
    """
    Plot historical decomposition.
    
    Parameters
    ----------
    x : bgvar.hd
        HD object from hd() function.
    var_slct : list, optional
        Variables to plot.
    **kwargs
        Additional plotting arguments.
        
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    HD = x.get('HD', None)
    if HD is None:
        raise ValueError("No HD data available for plotting.")
    
    varNames = list(x['xglobal'].columns)
    
    if var_slct is None:
        var_slct = varNames
    
    bigT, bigK, _ = HD.shape
    
    # Create subplots
    n_vars = len(var_slct)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(var_slct):
        var_idx = varNames.index(var)
        ax = axes[i]
        
        # Stacked area plot of contributions
        hd_data = HD[:, var_idx, :].T  # (K x T)
        
        ax.stackplot(range(bigT), *hd_data, labels=[f'Shock {j}' for j in range(bigK)], alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Contribution')
        ax.set_title(f'Historical Decomposition: {var}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pred(x,
             var_slct: Optional[List[str]] = None,
             **kwargs) -> plt.Figure:
    """
    Plot predictions.
    
    Parameters
    ----------
    x : bgvar.pred
        Prediction object from predict() function.
    var_slct : list, optional
        Variables to plot.
    **kwargs
        Additional plotting arguments.
        
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    fcast = x.get('fcast', None)
    if fcast is None:
        raise ValueError("No forecast data available for plotting.")
    
    xglobal = x.get('xglobal', None)
    if xglobal is None:
        raise ValueError("No data available for plotting.")
    
    varNames = list(xglobal.columns)
    
    if var_slct is None:
        var_slct = varNames
    
    bigK, n_ahead, Q = fcast.shape
    
    # Create subplots
    n_vars = len(var_slct)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(var_slct):
        var_idx = varNames.index(var)
        ax = axes[i]
        
        # Plot historical data
        hist_data = xglobal[var].values
        ax.plot(hist_data, label='Historical', color='black', linewidth=2)
        
        # Plot forecasts with credible intervals
        forecast_start = len(hist_data)
        forecast_idx = np.arange(forecast_start, forecast_start + n_ahead)
        
        # Median forecast
        median_idx = np.argmin(np.abs(np.array([0.05, 0.10, 0.16, 0.50, 0.84, 0.90, 0.95]) - 0.50))
        if median_idx < Q:
            ax.plot(forecast_idx, fcast[var_idx, :, median_idx], 
                   label='Forecast (median)', color='blue', linewidth=2)
        
        # Credible intervals (simplified - would use multiple quantiles)
        if Q >= 3:
            ax.fill_between(forecast_idx,
                          fcast[var_idx, :, 0],  # Lower bound
                          fcast[var_idx, :, -1],  # Upper bound
                          alpha=0.3, color='blue', label='90% CI')
        
        ax.axvline(forecast_start - 1, color='red', linestyle='--', linewidth=1, label='Forecast start')
        ax.set_xlabel('Time')
        ax.set_ylabel(var)
        ax.set_title(f'Forecast: {var}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

