"""
Utility functions for pyBGVAR package
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional
import warnings
import os


def get_shockinfo(ident: str = "chol", nr_rows: int = 1) -> pd.DataFrame:
    """
    Create a shockinfo dataframe for IRF analysis.
    
    Parameters
    ----------
    ident : str, optional
        Identification scheme. One of "chol", "girf", or "sign". Default is "chol".
    nr_rows : int, optional
        Number of rows (shocks) to initialize. Default is 1.
    
    Returns
    -------
    pd.DataFrame
        Shockinfo dataframe with appropriate columns for the identification scheme.
    """
    if ident == "chol":
        df = pd.DataFrame({
            'shock': [None] * nr_rows,
            'scale': [1.0] * nr_rows,
            'global': [False] * nr_rows
        })
        df.attrs['ident'] = 'chol'
    elif ident == "girf":
        df = pd.DataFrame({
            'shock': [None] * nr_rows,
            'scale': [1.0] * nr_rows,
            'global': [False] * nr_rows
        })
        df.attrs['ident'] = 'girf'
    elif ident == "sign":
        df = pd.DataFrame({
            'shock': [None] * nr_rows,
            'restriction': [None] * nr_rows,
            'sign': [None] * nr_rows,
            'horizon': [None] * nr_rows,
            'scale': [None] * nr_rows,
            'prob': [None] * nr_rows,
            'global': [None] * nr_rows
        })
        df.attrs['ident'] = 'sign'
    else:
        raise ValueError(f"Unknown identification scheme: {ident}")
    
    return df


def add_shockinfo(shockinfo: Optional[pd.DataFrame] = None, 
                 shock: Optional[str] = None,
                 restriction: Optional[List[str]] = None, 
                 sign: Optional[List[str]] = None,
                 horizon: Optional[Union[int, List[int]]] = None, 
                 prob: Optional[float] = None,
                 scale: Optional[float] = None, 
                 global_shock: Optional[bool] = None,
                 horizon_fillup: bool = True) -> pd.DataFrame:
    """
    Add shocks to shockinfo dataframe for IRF analysis with sign restrictions.
    
    Parameters
    ----------
    shockinfo : pd.DataFrame, optional
        Existing shockinfo dataframe. If None, a new sign restriction dataframe is created.
    shock : str, optional
        Variable of interest for structural shock.
    restriction : list of str, optional
        Variables that are supposed to be sign restricted.
    sign : list of str, optional
        Signs for restrictions (e.g., ['+', '-', '+']).
    horizon : int or list of int, optional
        Horizons to which restriction should hold.
    prob : float, optional
        Probability with which restriction is supposed to hold (between 0 and 1).
    scale : float, optional
        Scaling parameter.
    global_shock : bool, optional
        If True, shock is defined as global shock.
    horizon_fillup : bool, optional
        If True, horizon is filled up to given horizon. Otherwise just one specific horizon.
    
    Returns
    -------
    pd.DataFrame
        Updated shockinfo dataframe.
    """
    if shockinfo is None:
        shockinfo = get_shockinfo(ident="sign")
    
    if shock is None:
        raise ValueError("Please specify structural shock. This corresponds to the variable the shock is originating from.")
    
    if isinstance(shock, list) and len(shock) > 1:
        raise ValueError("Please only specify one structural shock at once.")
    
    if restriction is None or sign is None:
        raise ValueError("Please specify 'restriction' together with 'sign'.")
    
    if len(restriction) != len(sign):
        raise ValueError("Please provide the arguments 'restriction' and 'sign' with equal length.")
    
    # Handle horizon
    if horizon is not None:
        if isinstance(horizon, int):
            horizon = [horizon]
        
        if len(restriction) != len(horizon):
            if len(horizon) != 1:
                raise ValueError("Please provide the argument 'horizon' either with length equal to one for all shocks or with an equal length of the restrictions.")
            horizon = horizon * len(restriction)
    else:
        horizon = [None] * len(restriction)
    
    # Build new rows
    new_rows = []
    for i in range(len(restriction)):
        if horizon_fillup and horizon[i] is not None:
            for h in range(horizon[i] + 1):
                row = {
                    'shock': shock if isinstance(shock, str) else shock[0],
                    'restriction': restriction[i],
                    'sign': sign[i],
                    'horizon': h,
                    'scale': scale,
                    'prob': prob,
                    'global': global_shock
                }
                new_rows.append(row)
        else:
            row = {
                'shock': shock if isinstance(shock, str) else shock[0],
                'restriction': restriction[i],
                'sign': sign[i],
                'horizon': horizon[i],
                'scale': scale,
                'prob': prob,
                'global': global_shock
            }
            new_rows.append(row)
    
    # Append to shockinfo
    new_df = pd.DataFrame(new_rows)
    
    # Remove empty first row if it exists
    if shockinfo.iloc[0].isnull().all():
        shockinfo = shockinfo.iloc[1:].reset_index(drop=True)
    
    shockinfo = pd.concat([shockinfo, new_df], ignore_index=True)
    shockinfo.attrs['ident'] = 'sign'
    
    return shockinfo


def matrix_to_list(datamat: Union[np.ndarray, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Convert input matrix to list format for BGVAR.
    
    Converts a matrix where columns are named as 'COUNTRY.VARIABLE' 
    (e.g., 'US.y', 'US.Dp') into a dictionary where keys are country 
    names and values are DataFrames with variable columns.
    
    Parameters
    ----------
    datamat : array-like or DataFrame
        Matrix of size T x K, where T is time periods and K is total 
        number of variables. Column names should be in format 'COUNTRY.VARIABLE'.
        
    Returns
    -------
    dict
        Dictionary of length N (number of countries), where each entry 
        is a DataFrame of size T x k_i (variables per country).
        
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'US.y': [1, 2, 3], 'US.Dp': [0.5, 0.6, 0.7],
    ...                      'EA.y': [1.1, 2.1, 3.1], 'EA.Dp': [0.4, 0.5, 0.6]})
    >>> data_dict = matrix_to_list(data)
    >>> print(list(data_dict.keys()))
    ['US', 'EA']
    """
    if isinstance(datamat, pd.DataFrame):
        df = datamat.copy()
    else:
        df = pd.DataFrame(datamat)
    
    if df.isna().any().any():
        raise ValueError("The data you have submitted contains NaNs. Please check the data.")
    
    colnames = df.columns
    
    # Check column naming convention
    if not all('.' in str(col) for col in colnames):
        raise ValueError("Please separate country- and variable names with a dot (e.g., 'US.y').")
    
    # Extract country names
    cN = list(set([str(col).split('.')[0] for col in colnames]))
    N = len(cN)
    
    if not all(len(c) >= 2 for c in cN):
        raise ValueError("Please provide entity names with at least two characters.")
    
    # Create dictionary
    datalist = {}
    for cc in cN:
        country_cols = [col for col in colnames if str(col).startswith(cc + '.')]
        if len(country_cols) == 0:
            continue
        country_data = df[country_cols].copy()
        # Remove country prefix from column names
        country_data.columns = [col.split('.', 1)[1] for col in country_cols]
        datalist[cc] = country_data
    
    if len(datalist) != N:
        warnings.warn(f"Some countries may not have data. Expected {N} countries, got {len(datalist)}.")
    
    return datalist


def list_to_matrix(datalist: Dict[str, Union[np.ndarray, pd.DataFrame]]) -> pd.DataFrame:
    """
    Convert input list to matrix format.
    
    Converts a dictionary where keys are country names and values are 
    DataFrames/matrices with variables as columns into a single matrix 
    with columns named as 'COUNTRY.VARIABLE'.
    
    Parameters
    ----------
    datalist : dict
        Dictionary of length N, where each entry is a DataFrame/matrix 
        of size T x k (variables per country).
        
    Returns
    -------
    DataFrame
        Matrix of size T x K (time periods x total variables).
        
    Examples
    --------
    >>> data_dict = {
    ...     'US': pd.DataFrame({'y': [1, 2, 3], 'Dp': [0.5, 0.6, 0.7]}),
    ...     'EA': pd.DataFrame({'y': [1.1, 2.1, 3.1], 'Dp': [0.4, 0.5, 0.6]})
    ... }
    >>> data_matrix = list_to_matrix(data_dict)
    >>> print(data_matrix.columns.tolist())
    ['US.y', 'US.Dp', 'EA.y', 'EA.Dp']
    """
    if not all(len(name) >= 2 for name in datalist.keys()):
        raise ValueError("Please provide entity names with at least two characters.")
    
    cN = list(datalist.keys())
    N = len(cN)
    
    # Check for NaNs
    for name, data in datalist.items():
        if isinstance(data, pd.DataFrame):
            if data.isna().any().any():
                raise ValueError(f"The data for {name} contains NaNs. Please check the data.")
        else:
            if np.isnan(data).any():
                raise ValueError(f"The data for {name} contains NaNs. Please check the data.")
    
    # Combine all data
    datamat = None
    colnames = []
    
    for i, name in enumerate(cN):
        data = datalist[name]
        if isinstance(data, pd.DataFrame):
            country_df = data.copy()
        else:
            country_df = pd.DataFrame(data)
            if country_df.columns.empty:
                country_df.columns = [f'var{j}' for j in range(country_df.shape[1])]
        
        if datamat is None:
            datamat = country_df.copy()
        else:
            datamat = pd.concat([datamat, country_df], axis=1)
        
        # Create column names with country prefix
        for col in country_df.columns:
            colnames.append(f"{name}.{col}")
    
    datamat.columns = colnames
    
    return datamat


def mlag(X: Union[np.ndarray, pd.DataFrame], lag: int) -> pd.DataFrame:
    """
    Create lagged variables.
    
    Parameters
    ----------
    X : array-like or DataFrame
        Input data of size T x N (time x variables).
    lag : int
        Number of lags.
        
    Returns
    -------
    DataFrame
        Lagged data of size T x (N*lag).
        
    Examples
    --------
    >>> data = pd.DataFrame({'y': [1, 2, 3, 4, 5], 'x': [0.5, 0.6, 0.7, 0.8, 0.9]})
    >>> lagged = mlag(data, lag=2)
    >>> print(lagged.shape)
    (5, 4)
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        colnames = X.columns
    else:
        X_array = np.asarray(X)
        colnames = [f'var{i}' for i in range(X_array.shape[1])]
    
    Traw, N = X_array.shape
    p = lag
    
    # Initialize lagged matrix
    Xlag = np.zeros((Traw, p * N))
    
    # Create lags
    lag_colnames = []
    for ii in range(1, p + 1):
        start_idx = N * (ii - 1)
        end_idx = N * ii
        Xlag[(p + 1 - ii):(Traw - ii + 1), start_idx:end_idx] = \
            X_array[(p + 1 - ii):(Traw - ii + 1), :]
        lag_colnames.extend([f"{col}.lag{ii}" for col in colnames])
    
    Xlag_df = pd.DataFrame(Xlag, columns=lag_colnames)
    
    return Xlag_df


def check_data_format(data: Union[Dict, pd.DataFrame, np.ndarray]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Check and normalize data format for BGVAR.
    
    Parameters
    ----------
    data : dict, DataFrame, or array
        Input data in various formats.
        
    Returns
    -------
    tuple
        (data_dict, country_names) where data_dict is normalized dictionary
        and country_names is list of country codes.
    """
    if isinstance(data, dict):
        # Already in dictionary format
        data_dict = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                data_dict[key] = value.copy()
            else:
                data_dict[key] = pd.DataFrame(value)
        
        cN = list(data_dict.keys())
        
        # Check that all have same number of rows
        T_values = [df.shape[0] for df in data_dict.values()]
        if len(set(T_values)) > 1:
            raise ValueError("Please provide same sample size for all countries.")
        
        return data_dict, cN
    
    elif isinstance(data, (pd.DataFrame, np.ndarray)):
        # Convert matrix to list
        data_dict = matrix_to_list(data)
        cN = list(data_dict.keys())
        return data_dict, cN
    
    else:
        raise TypeError("Please provide the argument 'data' either as 'dict' or as 'DataFrame/array'.")


def check_weight_matrix(W: Union[Dict, np.ndarray, pd.DataFrame], 
                        cN: List[str], 
                        OE_weights: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """
    Check and normalize weight matrix format.
    
    Parameters
    ----------
    W : dict, array, or DataFrame
        Weight matrix or dictionary of weight matrices.
    cN : list
        List of country names.
    OE_weights : dict, optional
        Dictionary for other entities weights.
        
    Returns
    -------
    dict
        Dictionary of weight matrices.
    """
    if isinstance(W, dict):
        W_dict = {}
        for key, value in W.items():
            if isinstance(value, (pd.DataFrame, np.ndarray)):
                W_dict[key] = np.asarray(value)
            else:
                raise TypeError(f"Weight matrix {key} must be array or DataFrame.")
        return W_dict
    
    elif isinstance(W, (np.ndarray, pd.DataFrame)):
        # Convert single matrix to dictionary
        W_array = np.asarray(W)
        if W_array.shape[0] != W_array.shape[1]:
            raise ValueError("Weight matrix must be square.")
        
        if OE_weights is None:
            if W_array.shape[0] != len(cN):
                raise ValueError(f"Data and W matrix not of the same dimension. "
                               f"Data has {len(cN)} countries, W has {W_array.shape[0]} rows.")
        
        return {'W': W_array}
    
    else:
        raise TypeError("Please provide the argument 'W' either as 'dict' or as 'array/DataFrame'.")


def excel_to_list(file: str, 
                  first_column_as_time: bool = True, 
                  skipsheet: Optional[Union[List[str], List[int]]] = None,
                  **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Read data from Excel file and convert to dictionary format.
    
    Reads an Excel spreadsheet where each sheet represents data for one country.
    Column names are used as variable names. If `first_column_as_time` is True,
    the first column is treated as a time index.
    
    Parameters
    ----------
    file : str
        Path to the Excel file (.xls or .xlsx).
    first_column_as_time : bool, optional
        Whether the first column in each sheet represents time. Default is True.
    skipsheet : list, optional
        List of sheet names (strings) or indices (integers) to skip.
        Default is None (read all sheets).
    **kwargs
        Additional arguments passed to pandas.read_excel().
    
    Returns
    -------
    dict
        Dictionary where keys are country names (sheet names) and values
        are DataFrames with time as index (if first_column_as_time=True)
        or regular DataFrames.
    
    Examples
    --------
    >>> data_dict = excel_to_list('data.xlsx', first_column_as_time=True)
    >>> # Each sheet represents a country, e.g., 'US', 'EA'
    >>> # First column is time index, remaining columns are variables
    """
    # Check if file exists
    if not os.path.exists(file):
        raise FileNotFoundError(f"The provided file does not exist: {file}")
    
    # Check file extension
    if not file.lower().endswith(('.xls', '.xlsx')):
        raise ValueError("Please provide a path to an Excel file (ending with .xls or .xlsx).")
    
    # Read all sheet names
    try:
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}")
    
    # Filter out skipped sheets
    if skipsheet is not None:
        if len(skipsheet) > 0:
            if isinstance(skipsheet[0], str):
                # Skip by name
                sheet_names = [name for name in sheet_names if name not in skipsheet]
            elif isinstance(skipsheet[0], int):
                # Skip by index (1-based to 0-based conversion)
                sheet_names = [name for i, name in enumerate(sheet_names) 
                              if (i + 1) not in skipsheet]
    
    # Read each sheet
    datalist = {}
    for sheet_name in sheet_names:
        try:
            # Read the sheet
            df = pd.read_excel(file, sheet_name=sheet_name, **kwargs)
            
            if first_column_as_time:
                # First column is time index
                if df.empty:
                    warnings.warn(f"Sheet '{sheet_name}' is empty. Skipping.")
                    continue
                
                # Check if first column exists and is time-like
                time_col = df.iloc[:, 0]
                
                # Try to convert to datetime if it's not already
                try:
                    time_index = pd.to_datetime(time_col)
                except:
                    # If conversion fails, use as string
                    time_index = pd.Index(time_col.astype(str))
                
                # Use remaining columns as data
                data = df.iloc[:, 1:].copy()
                data.index = time_index
                data.columns.name = None  # Remove column name
                
            else:
                # No time column, use all columns as data
                data = df.copy()
            
            # Convert to numeric (handle any non-numeric columns)
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with all NaN
            data = data.dropna(how='all')
            
            if data.empty:
                warnings.warn(f"Sheet '{sheet_name}' contains no valid data. Skipping.")
                continue
            
            datalist[sheet_name] = data
            
        except Exception as e:
            warnings.warn(f"Error reading sheet '{sheet_name}': {e}. Skipping.")
            continue
    
    if len(datalist) == 0:
        raise ValueError("No valid data found in Excel file. Please check the file format.")
    
    return datalist

