"""
pyBGVAR: Python implementation of Bayesian Global Vector Autoregressions

This package provides a Python implementation of Bayesian Global Vector 
Autoregression models, originally implemented in R (BGVAR package).
"""

__version__ = "0.1.0"
__author__ = "Python BGVAR Team"

from .bgvar import BGVAR
from . import utils
from . import helpers
from . import irf
from . import fevd
from . import hd
from . import predict
from . import plot
from . import diagnostics
from . import bvar
from . import stacking

# Utility functions
from .utils import (
    get_shockinfo,
    add_shockinfo,
    matrix_to_list,
    list_to_matrix,
    excel_to_list
)

# Diagnostics functions
from .diagnostics import (
    conv_diag,
    resid_corr_test,
    avg_pair_cc
)

# Prediction evaluation functions
from .predict import (
    lps,
    rmse
)

# FEVD functions
from .fevd import (
    fevd as compute_fevd,
    gfevd
)

__all__ = [
    # Main class
    "BGVAR",
    
    # Modules
    "utils",
    "helpers",
    "irf",
    "fevd",
    "hd",
    "predict",
    "plot",
    "diagnostics",
    "bvar",
    "stacking",
    
    # Utility functions
    "get_shockinfo",
    "add_shockinfo",
    "matrix_to_list",
    "list_to_matrix",
    "excel_to_list",
    
    # Diagnostics
    "conv_diag",
    "resid_corr_test",
    "avg_pair_cc",
    
    # Prediction evaluation
    "lps",
    "rmse",
    
    # FEVD
    "compute_fevd",
    "gfevd",
]

