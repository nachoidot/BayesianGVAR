"""
Helper functions for BGVAR estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.linalg import cholesky, solve, inv
from scipy.stats import norm, gamma


def get_weights(Data: Dict[str, pd.DataFrame],
                W: Dict[str, np.ndarray],
                OE_weights: Optional[Dict] = None,
                Wex_restr: Optional[List[str]] = None,
                variable_list: Optional[Dict] = None) -> Dict:
    """
    Construct weight matrices for each country model.
    
    This function processes the data and weight matrices to create
    country-specific weight matrices (gW) for GVAR estimation.
    
    Parameters
    ----------
    Data : dict
        Dictionary of country DataFrames.
    W : dict
        Dictionary of weight matrices.
    OE_weights : dict, optional
        Dictionary for other entities weights.
    Wex_restr : list, optional
        List of variables to exclude from weakly exogenous variables.
    variable_list : dict, optional
        Mapping of variables to weight matrices when multiple W matrices are provided.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'gW': Dictionary of country-specific weight matrices
        - 'bigx': Global data matrix
        - 'exo': Dictionary of exogenous variable indices
        - 'exo.countries': List of countries with exogenous variables
        - 'endo': Dictionary of endogenous variable indices
    """
    cN = list(Data.keys())
    N = len(cN)
    
    # Get variable names
    nn = {cc: list(Data[cc].columns) for cc in cN}
    
    # Identify exogenous variables (variables that appear in only one country)
    all_vars = []
    for vars_list in nn.values():
        all_vars.extend(vars_list)
    
    var_counts = {}
    for var in all_vars:
        var_counts[var] = var_counts.get(var, 0) + 1
    
    # Exogenous variables are those that appear only once
    exo_vars = [var for var, count in var_counts.items() if count == 1]
    endo_vars = [var for var, count in var_counts.items() if count > 1]
    
    # Find countries with exogenous variables
    exo_countries = []
    for cc in cN:
        if any(var in nn[cc] for var in exo_vars):
            exo_countries.append(cc)
    
    # Build global data matrix
    bigx_list = []
    for cc in cN:
        country_data = Data[cc].values
        var_names = Data[cc].columns
        country_cols = [f"{cc}.{var}" for var in var_names]
        country_df = pd.DataFrame(country_data, columns=country_cols)
        bigx_list.append(country_df)
    
    bigx = pd.concat(bigx_list, axis=1)
    
    # Remove duplicates (in case of overlapping variable names)
    bigx = bigx.loc[:, ~bigx.columns.duplicated()]
    
    # Process weight matrices for each country
    gW = {}
    max_char = max([len(col) for col in bigx.columns])
    cnt_char = max([len(cc) for cc in cN])
    
    # Determine which weight matrix to use for which variables
    if variable_list is None:
        variable_list = {'vars': endo_vars}
    
    # If multiple weight matrices, check variable_list matches
    if len(W) > 1:
        if len(W) != len(variable_list):
            raise ValueError("Number of weight matrices must match number of variable sets.")
        if not all(var in variable_list.values() for var in endo_vars):
            raise ValueError("All endogenous variables must be assigned to a weight matrix.")
    
    # For each country, build the weight matrix
    for cc_idx, cc in enumerate(cN):
        # Get country-specific variables
        country_vars = [var for var in nn[cc]]
        all_var_names = list(set([col.split('.', 1)[1] for col in bigx.columns]))
        
        # Apply Wex_restr if provided
        if Wex_restr is not None:
            all_var_names = [v for v in all_var_names if v not in Wex_restr]
        
        # Initialize weight matrix for this country
        Wnew = np.zeros((len(all_var_names), bigx.shape[1]))
        Wnew_df = pd.DataFrame(Wnew, index=all_var_names, columns=bigx.columns)
        
        # Get the appropriate weight matrix
        if len(W) == 1:
            W_matrix = list(W.values())[0]
        else:
            # Need to determine which W to use based on variable_list
            # For simplicity, use first one if multiple - this should be refined
            W_matrix = list(W.values())[0]
        
        # Fill in weights for weakly exogenous variables
        for var_idx, var_name in enumerate(all_var_names):
            # Find columns in bigx that match this variable
            matching_cols = [col for col in bigx.columns if col.split('.', 1)[1] == var_name]
            
            if len(matching_cols) > 0:
                # Extract country codes
                countries_with_var = [col.split('.', 1)[0] for col in matching_cols]
                
                if cc in countries_with_var:
                    # This is an endogenous variable for this country
                    # Set diagonal element to 1
                    own_col = f"{cc}.{var_name}"
                    if own_col in bigx.columns:
                        Wnew_df.loc[var_name, own_col] = 1.0
                    
                    # Set weights for other countries' versions of this variable
                    for other_country in countries_with_var:
                        if other_country != cc:
                            other_col = f"{other_country}.{var_name}"
                            if other_col in bigx.columns:
                                # Get weight from W matrix
                                try:
                                    weight = W_matrix[cc_idx, list(cN).index(other_country)]
                                    Wnew_df.loc[var_name, other_col] = weight
                                except (IndexError, ValueError):
                                    Wnew_df.loc[var_name, other_col] = 0.0
                else:
                    # This is an exogenous variable
                    # Set weights based on W or equal weights
                    for col in matching_cols:
                        Wnew_df.loc[var_name, col] = 1.0 / len(matching_cols)
            
            # Handle exogenous countries
            if cc in exo_countries:
                own_col = f"{cc}.{var_name}"
                if own_col in bigx.columns:
                    Wnew_df.loc[var_name, own_col] = 1.0
        
        # Normalize rows to sum to 1
        Wnew_array = Wnew_df.values
        row_sums = Wnew_array.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        Wnew_array = Wnew_array / row_sums
        
        # Remove zero rows
        non_zero_rows = (Wnew_array.sum(axis=1) != 0)
        Wnew_array = Wnew_array[non_zero_rows, :]
        Wnew_index = [Wnew_df.index[i] for i in range(len(Wnew_df.index)) if non_zero_rows[i]]
        
        # Build endogenous weight matrix
        country_var_names = [f"{cc}.{var}" for var in country_vars]
        endoW = np.zeros((len(country_var_names), bigx.shape[1]))
        
        for j, country_var in enumerate(country_var_names):
            if country_var in bigx.columns:
                var_idx_in_bigx = list(bigx.columns).index(country_var)
                endoW[j, var_idx_in_bigx] = 1.0
        
        # Combine endogenous and exogenous weight matrices
        WfinNR = np.vstack([endoW, Wnew_array])
        gW[cc] = WfinNR
    
    return {
        'gW': gW,
        'bigx': bigx,
        'exo': {var: 1 for var in exo_vars},
        'exo.countries': exo_countries,
        'endo': {var: 1 for var in endo_vars}
    }


def get_companion(Beta: np.ndarray, varndxv: List[int]) -> Dict[str, np.ndarray]:
    """
    Construct companion matrix from VAR coefficients.
    
    Parameters
    ----------
    Beta : array
        Coefficient matrix.
    varndxv : list
        [number of variables, number of deterministics, number of lags]
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'MM': Companion matrix
        - 'Jm': Selection matrix
    """
    nn = varndxv[0]  # number of variables
    nd = varndxv[1]  # number of deterministics
    nl = varndxv[2]  # number of lags
    
    nkk = nn * nl + nd
    
    # Selection matrix
    Jm = np.zeros((nkk, nn))
    Jm[:nn, :nn] = np.eye(nn)
    
    # Companion matrix
    if nd > 0:
        # With deterministics
        MM = np.zeros((nkk, nkk))
        MM[:Beta.shape[0], :Beta.shape[1]] = Beta
        
        # Identity blocks for lags
        for i in range(nl - 1):
            start_row = Beta.shape[0] + i * nn
            start_col = i * nn
            MM[start_row:start_row + nn, start_col:start_col + nn] = np.eye(nn)
        
        # Deterministic terms
        det_start = nn * nl
        MM[det_start:, det_start:] = np.eye(nd)
    else:
        # Without deterministics
        MM = np.zeros((nkk, nkk))
        MM[:Beta.shape[0], :Beta.shape[1]] = Beta
        
        # Identity blocks for lags
        for i in range(nl - 1):
            start_row = Beta.shape[0] + i * nn
            start_col = i * nn
            MM[start_row:start_row + nn, start_col:start_col + nn] = np.eye(nn)
    
    return {'MM': MM, 'Jm': Jm}


def get_V(k: int, M: int, Mstar: int, plag: int, plagstar: int,
          lambda1: float, lambda2: float, lambda3: float, lambda4: float,
          sigma_sq: np.ndarray, sigma_wex: Optional[np.ndarray] = None,
          trend: bool = False, wexo: bool = True) -> np.ndarray:
    """
    Construct Minnesota prior variance matrix.
    
    Parameters
    ----------
    k : int
        Total number of regressors.
    M : int
        Number of endogenous variables.
    Mstar : int
        Number of weakly exogenous variables.
    plag : int
        Number of lags for endogenous variables.
    plagstar : int
        Number of lags for weakly exogenous variables.
    lambda1, lambda2, lambda3, lambda4 : float
        Prior hyperparameters.
    sigma_sq : array
        Residual variances for endogenous variables.
    sigma_wex : array, optional
        Residual variances for weakly exogenous variables.
    trend : bool
        Whether trend is included.
    wexo : bool
        Whether weakly exogenous variables are present.
        
    Returns
    -------
    array
        Prior variance matrix of size k x M.
    """
    V_i = np.zeros((k, M))
    
    # Endogenous part
    for i in range(M):
        for pp in range(1, plag + 1):
            for j in range(M):
                if i == j:
                    V_i[j + M * (pp - 1), i] = (lambda1 / pp) ** 2
                else:
                    V_i[j + M * (pp - 1), i] = ((lambda1 * lambda2 / pp) ** 2) * \
                                               (sigma_sq[i] / sigma_sq[j])
    
    # Exogenous part
    if wexo and Mstar > 0:
        if sigma_wex is None:
            sigma_wex = np.ones(Mstar)
        
        for i in range(M):
            for pp in range(0, plagstar + 1):
                for j in range(Mstar):
                    V_i[M * plag + pp * Mstar + j, i] = \
                        ((lambda1 * lambda3 / (pp + 1)) ** 2) * \
                        (sigma_sq[i] / sigma_wex[j])
    
    # Deterministics
    for i in range(M):
        if trend:
            V_i[k - 2:k, i] = lambda4 * sigma_sq[i]
        else:
            V_i[k - 1, i] = lambda4 * sigma_sq[i]
    
    return V_i

