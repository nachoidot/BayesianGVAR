"""
Main BGVAR estimation module
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple
import warnings
from datetime import datetime

from . import utils
from . import helpers


class BGVAR:
    """
    Bayesian Global Vector Autoregression Model
    
    This class implements Bayesian GVAR estimation with various prior setups:
    - Minnesota (MN)
    - Stochastic Search Variable Selection (SSVS)
    - Normal-Gamma (NG)
    - Horseshoe (HS)
    
    All specifications can be estimated with stochastic volatility.
    
    Parameters
    ----------
    data : dict or DataFrame
        Either a dictionary of length N (countries) containing DataFrames,
        or a DataFrame with columns named as 'COUNTRY.VARIABLE'.
    W : dict or array
        Weight matrix (N x N) or dictionary of weight matrices.
    plag : int or list, default=1
        Number of lags. Either single value or [endogenous_lags, exogenous_lags].
    draws : int, default=5000
        Number of retained MCMC draws.
    burnin : int, default=5000
        Number of burn-in iterations.
    prior : str, default='NG'
        Prior specification: 'MN', 'SSVS', 'NG', or 'HS'.
    SV : bool, default=True
        Whether to include stochastic volatility.
    hold_out : int, default=0
        Number of observations to hold out for validation.
    thin : int, default=1
        Thinning interval for MCMC chain.
    hyperpara : dict, optional
        Dictionary of hyperparameters for the prior.
    eigen : bool or float, default=True
        Whether to check eigenvalue stability. If float, maximum eigenvalue threshold.
    Ex : dict or DataFrame, optional
        Truly exogenous variables.
    trend : bool, default=False
        Whether to include deterministic trend.
    expert : dict, optional
        Expert settings for advanced options.
    verbose : bool, default=True
        Whether to print progress messages.
        
    Attributes
    ----------
    args : dict
        Estimation arguments.
    xglobal : DataFrame
        Global data matrix.
    gW : dict
        Country-specific weight matrices.
    stacked_results : dict
        Stacked global model results.
    cc_results : dict
        Country model results.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from pyBGVAR import BGVAR
    >>> 
    >>> # Prepare data
    >>> data = {
    ...     'US': pd.DataFrame({'y': [1, 2, 3], 'Dp': [0.5, 0.6, 0.7]}),
    ...     'EA': pd.DataFrame({'y': [1.1, 2.1, 3.1], 'Dp': [0.4, 0.5, 0.6]})
    ... }
    >>> W = np.array([[0.0, 1.0], [1.0, 0.0]])  # Simple weight matrix
    >>> 
    >>> # Estimate model
    >>> model = BGVAR(
    ...     data=data,
    ...     W=W,
    ...     plag=1,
    ...     draws=100,
    ...     burnin=100,
    ...     prior='NG',
    ...     SV=False
    ... )
    """
    
    def __init__(self,
                 data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                 W: Union[Dict[str, np.ndarray], np.ndarray],
                 plag: Union[int, List[int]] = 1,
                 draws: int = 5000,
                 burnin: int = 5000,
                 prior: str = 'NG',
                 SV: bool = True,
                 hold_out: int = 0,
                 thin: int = 1,
                 hyperpara: Optional[Dict] = None,
                 eigen: Union[bool, float] = True,
                 Ex: Optional[Union[Dict[str, pd.DataFrame], pd.DataFrame]] = None,
                 trend: bool = False,
                 expert: Optional[Dict] = None,
                 verbose: bool = True):
        
        self.start_time = datetime.now()
        
        # Store all arguments
        self.args = {
            'data': data,
            'W': W,
            'plag': plag,
            'draws': draws,
            'burnin': burnin,
            'prior': prior,
            'SV': SV,
            'hold_out': hold_out,
            'thin': thin,
            'hyperpara': hyperpara,
            'eigen': eigen,
            'Ex': Ex,
            'trend': trend,
            'expert': expert,
            'verbose': verbose
        }
        
        # Validate inputs
        self._validate_inputs()
        
        # Process data
        self._process_data()
        
        # Process weight matrices
        self._process_weights()
        
        # Set hyperparameters
        self._set_hyperparameters()
        
        # Process expert settings
        self._process_expert_settings()
        
        # Print initialization message
        if verbose:
            self._print_init_message()
        
        # Estimate model
        self._estimate()
        
        if verbose:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            print(f"\nTotal estimation time: {elapsed:.2f} seconds")
    
    def _validate_inputs(self):
        """Validate input arguments."""
        # Check data format
        if not isinstance(self.args['data'], (dict, pd.DataFrame, np.ndarray)):
            raise TypeError("'data' must be dict, DataFrame, or array.")
        
        # Check W format
        if not isinstance(self.args['W'], (dict, np.ndarray, pd.DataFrame)):
            raise TypeError("'W' must be dict, array, or DataFrame.")
        
        # Check lag specification
        plag = self.args['plag']
        if isinstance(plag, (list, tuple, np.ndarray)):
            if len(plag) != 2:
                raise ValueError("If 'plag' is a list, it must have length 2.")
            self.args['lags'] = list(plag)
        elif isinstance(plag, (int, np.integer)):
            self.args['lags'] = [int(plag), int(plag)]
        else:
            raise TypeError("'plag' must be int or list of two ints.")
        
        # Check prior
        valid_priors = ['MN', 'SSVS', 'NG', 'HS']
        if self.args['prior'] not in valid_priors:
            raise ValueError(f"'prior' must be one of {valid_priors}.")
        
        # Check draws and burnin
        if not isinstance(self.args['draws'], (int, np.integer)) or self.args['draws'] < 1:
            raise ValueError("'draws' must be a positive integer.")
        if not isinstance(self.args['burnin'], (int, np.integer)) or self.args['burnin'] < 0:
            raise ValueError("'burnin' must be a non-negative integer.")
        
        # Check thin
        if self.args['thin'] < 1:
            warnings.warn(f"Thinning factor {self.args['thin']} adjusted to {1/self.args['thin']:.2f}.")
            self.args['thin'] = round(1 / self.args['thin'], 2)
        
        # Check thin divides draws
        if self.args['draws'] % self.args['thin'] != 0:
            divisors = [d for d in range(1, self.args['draws'] + 1) 
                       if self.args['draws'] % d == 0]
            closest = min(divisors, key=lambda x: abs(x - self.args['thin']))
            warnings.warn(f"Thinning factor {self.args['thin']} adjusted to {closest}.")
            self.args['thin'] = closest
    
    def _process_data(self):
        """Process and validate data."""
        data = self.args['data']
        
        # Convert to dictionary format
        if isinstance(data, dict):
            self.data_dict, self.cN = utils.check_data_format(data)
        else:
            self.data_dict, self.cN = utils.check_data_format(data)
        
        self.N = len(self.cN)
        
        # Check country names are exactly 2 characters (following R convention)
        if not all(len(c) == 2 for c in self.cN):
            raise ValueError("Please provide entity names with exactly two characters.")
        
        # Check for missing values
        for cc, df in self.data_dict.items():
            if df.isna().any().any():
                raise ValueError(f"Data for {cc} contains NaNs.")
            if df.shape[1] < 2:
                raise ValueError(f"Each country must have at least 2 variables. {cc} has {df.shape[1]}.")
        
        # Create global data matrix
        data_list = []
        for cc in self.cN:
            country_data = self.data_dict[cc]
            country_cols = [f"{cc}.{var}" for var in country_data.columns]
            country_df = pd.DataFrame(
                country_data.values,
                columns=country_cols,
                index=country_data.index if hasattr(country_data, 'index') else None
            )
            data_list.append(country_df)
        
        self.xglobal = pd.concat(data_list, axis=1)
        self.args['Traw'] = self.xglobal.shape[0]
        self.args['time'] = list(range(self.args['Traw']))
        
        # Remove hold-out sample if specified
        if self.args['hold_out'] > 0:
            self.xglobal = self.xglobal.iloc[:-self.args['hold_out']]
            self.args['Traw'] = self.xglobal.shape[0]
    
    def _process_weights(self):
        """Process weight matrices."""
        W = self.args['W']
        expert = self.args.get('expert')
        OE_weights = expert.get('OE.weights', None) if expert else None
        
        self.W_dict = utils.check_weight_matrix(W, self.cN, OE_weights)
        
        # Validate weight matrices
        for key, W_matrix in self.W_dict.items():
            if W_matrix.shape[0] != W_matrix.shape[1]:
                raise ValueError(f"Weight matrix {key} must be square.")
            if OE_weights is None:
                if W_matrix.shape[0] != self.N:
                    raise ValueError(f"Data and W matrix {key} not of the same dimension.")
            
            # Check row sums (should be approximately 1)
            row_sums = W_matrix.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                warnings.warn(f"Weight matrix {key} row sums are not exactly 1. Normalizing...")
                W_matrix = W_matrix / row_sums[:, np.newaxis]
                self.W_dict[key] = W_matrix
    
    def _set_hyperparameters(self):
        """Set default hyperparameters and override with user-specified values."""
        # Default hyperparameters
        default_hyperpara = {
            'a_1': 3.0,
            'b_1': 0.3,
            'prmean': 0.0,
            'Bsigma': 1.0,
            'a0': 25.0,
            'b0': 1.5,
            'bmu': 0.0,
            'Bmu': 100.0**2,
            # Minnesota
            'lambda1': 0.1,
            'lambda2': 0.2,
            'lambda3': 0.1,
            'lambda4': 100.0,
            # SSVS
            'tau0': 0.1,
            'tau1': 3.0,
            'kappa0': 0.1,
            'kappa1': 7.0,
            'p_i': 0.5,
            'q_ij': 0.5,
            # NG
            'd_lambda': 0.01,
            'e_lambda': 0.01,
            'tau_theta': 0.7,
            'sample_tau': True,
            'tau_log': True
        }
        
        # Override with user-specified values
        user_hyperpara = self.args.get('hyperpara', {})
        if user_hyperpara:
            for key, value in user_hyperpara.items():
                if key in default_hyperpara:
                    default_hyperpara[key] = value
                else:
                    warnings.warn(f"Unknown hyperparameter: {key}. Ignoring.")
        
        self.hyperpara = default_hyperpara
    
    def _process_expert_settings(self):
        """Process expert settings."""
        expert = self.args.get('expert', {})
        
        default_expert = {
            'variable.list': None,
            'OE.weights': None,
            'Wex.restr': None,
            'save.country.store': False,
            'save.shrink.store': False,
            'save.vola.store': False,
            'use_R': False,
            'applyfun': None,
            'cores': None
        }
        
        for key, value in expert.items():
            if key in default_expert:
                default_expert[key] = value
        
        self.expert = default_expert
    
    def _print_init_message(self):
        """Print initialization message."""
        prior_names = {
            'MN': 'Minnesota prior',
            'SSVS': 'Stochastic Search Variable Selection prior',
            'NG': 'Normal-Gamma prior',
            'HS': 'Horseshoe prior'
        }
        
        print("\n" + "="*80)
        print("Start estimation of Bayesian Global Vector Autoregression")
        print("="*80)
        print(f"Prior: {prior_names.get(self.args['prior'], self.args['prior'])}")
        print(f"Lag order: {self.args['lags'][0]} (endo.), {self.args['lags'][1]} (w. exog.)")
        print(f"Stochastic volatility: {'enabled' if self.args['SV'] else 'disabled'}")
        print(f"Number of countries: {self.N}")
        print(f"Total variables: {self.xglobal.shape[1]}")
        print(f"Sample size: {self.args['Traw']}")
        print(f"Number of draws: {self.args['draws']}")
        print(f"Burn-in: {self.args['burnin']}")
        print(f"Thinning: {self.args['thin']}")
        print("="*80 + "\n")
    
    def _estimate(self):
        """Main estimation function."""
        # Get weights for each country model
        weight_result = helpers.get_weights(
            Data=self.data_dict,
            W=self.W_dict,
            OE_weights=self.expert.get('OE.weights'),
            Wex_restr=self.expert.get('Wex.restr'),
            variable_list=self.expert.get('variable.list')
        )
        
        self.gW = weight_result['gW']
        self.xglobal = weight_result['bigx']
        self.exo = weight_result['exo']
        self.exo_countries = weight_result['exo.countries']
        self.endo = weight_result['endo']
        
        # Estimate country models
        if self.args['verbose']:
            print("Estimation of country models starts...")
        
        # Import here to avoid circular imports
        from . import bvar
        from . import stacking
        
        setting_store = {
            'shrink_MN': self.expert.get('save.shrink.store', False) and self.args['prior'] == 'MN',
            'shrink_SSVS': self.expert.get('save.shrink.store', False) and self.args['prior'] == 'SSVS',
            'shrink_NG': self.expert.get('save.shrink.store', False) and self.args['prior'] == 'NG',
            'shrink_HS': self.expert.get('save.shrink.store', False) and self.args['prior'] == 'HS',
            'vola_pars': self.expert.get('save.vola.store', False)
        }
        
        self.globalpost = {}
        for cc in self.cN:
            if self.args['verbose']:
                print(f"Estimating model for {cc}...")
            
            # Extract country data
            Yraw = self.data_dict[cc].values
            W_matrix = self.gW[cc]
            
            # Compute weakly exogenous variables
            xglobal_array = self.xglobal.values
            Wraw = None
            if W_matrix.shape[1] > Yraw.shape[1]:
                all_vars = W_matrix @ xglobal_array.T
                Wraw = all_vars[:, Yraw.shape[1]:].T
            
            Exraw = None  # Placeholder for truly exogenous
            
            # Estimate BVAR
            bvar_result = bvar.estimate_bvar(
                Yraw=Yraw,
                Wraw=Wraw,
                Exraw=Exraw,
                lags=self.args['lags'],
                draws=self.args['draws'],
                burnin=self.args['burnin'],
                thin=self.args['thin'],
                cons=True,
                trend=self.args['trend'],
                sv=self.args['SV'],
                prior=self.args['prior'],
                hyperpara=self.hyperpara,
                setting_store=setting_store,
                verbose=self.args['verbose']
            )
            
            # Store with W matrix for stacking
            bvar_result['W'] = W_matrix
            self.globalpost[cc] = bvar_result
        
        # Stack global model
        if self.args['verbose']:
            print("\nStacking of global model starts...")
        
        self.stacked_results = stacking.stack_gvar(
            xglobal=self.xglobal,
            plag=max(self.args['lags']),
            globalpost=self.globalpost,
            draws=self.args['draws'],
            thin=self.args['thin'],
            trend=self.args['trend'],
            eigen=bool(self.args['eigen']),
            trim=self.args['eigen'] if isinstance(self.args['eigen'], (int, float)) else None,
            verbose=self.args['verbose']
        )
        
        # Prepare country results
        self.cc_results = self._prepare_country_results()
        
        if self.args['verbose']:
            stable_draws = len(self.stacked_results.get('F.eigen', []))
            print(f"\nEstimation finished. {stable_draws} stable draws retained.")
    
    def _prepare_country_results(self):
        """Prepare country model results summary."""
        cc_results = {
            'coeffs': {},
            'sig': {},
            'theta': {},
            'res': {}
        }
        
        # Extract posterior medians for each country
        for cc in self.cN:
            if cc not in self.globalpost:
                continue
            
            post = self.globalpost[cc]
            
            # Posterior median coefficients
            if 'A_store' in post:
                A_post = np.median(post['A_store'], axis=2)
                cc_results['coeffs'][cc] = A_post
            
            # Posterior median variance-covariance
            if 'L_store' in post and 'Sv_store' in post:
                L_post = np.median(post['L_store'], axis=2)
                Sv_post = np.median(post['Sv_store'], axis=2)
                # Construct SIGMA
                bigT = post['Y'].shape[0]
                M = post['Y'].shape[1]
                SIGMA = np.zeros((bigT, M, M))
                for t in range(bigT):
                    if M > 1:
                        SIGMA[t, :, :] = L_post @ np.diag(np.exp(Sv_post[t, :])) @ L_post.T
                    else:
                        SIGMA[t, 0, 0] = np.exp(Sv_post[t, 0])
                cc_results['sig'][cc] = np.median(SIGMA, axis=0)
            
            # Residuals
            if 'res_store' in post:
                res_post = np.median(post['res_store'], axis=2)
                cc_results['res'][cc] = res_post
        
        return cc_results
    
    def irf(self, n_ahead=24, shockinfo=None, quantiles=None, expert=None, verbose=True):
        """Compute impulse response functions."""
        from . import irf as irf_module
        return irf_module.irf(self, n_ahead, shockinfo, quantiles, expert, verbose)
    
    def fevd(self, irf_result, rotation_matrix=None, var_slct=None, verbose=True):
        """Compute forecast error variance decomposition."""
        from . import fevd as fevd_module
        return fevd_module.fevd(irf_result, rotation_matrix, var_slct, verbose)
    
    def hd(self, irf_result, var_slct=None, verbose=True):
        """Compute historical decomposition."""
        from . import hd as hd_module
        return hd_module.hd(irf_result, var_slct, verbose)
    
    def predict(self, n_ahead=1, constr=None, constr_sd=None, quantiles=None,
               save_store=False, verbose=True):
        """Compute predictions."""
        from . import predict as predict_module
        return predict_module.predict(self, n_ahead, constr, constr_sd, quantiles,
                                     save_store, verbose)
    
    def summary(self):
        """
        Generate a summary of the BGVAR model.
        
        Returns
        -------
        dict
            Dictionary containing model summary statistics including convergence diagnostics,
            residual correlation tests, and average pairwise cross-correlations.
        """
        from . import diagnostics
        
        if self.args.get('thindraws', 0) == 0:
            print("Computation of BGVAR has yielded no stable posterior draws!")
            return None
        
        CD = diagnostics.conv_diag(self)
        res = diagnostics.resid_corr_test(self, lag_cor=1, alpha=0.95)
        cross_corr = diagnostics.avg_pair_cc(self)
        
        summary_dict = {
            'object': self,
            'CD': CD,
            'res': res,
            'cross_corr': cross_corr
        }
        
        # Print summary
        print("-" * 75)
        print("Model Info:")
        prior_names = {
            'MN': 'Minnesota prior (MN)',
            'SSVS': 'Stochastic Search Variable Selection prior (SSVS)',
            'NG': 'Normal-Gamma prior (NG)',
            'HS': 'Horseshoe prior (HS)'
        }
        print(f"Prior: {prior_names.get(self.args['prior'], self.args['prior'])}")
        print(f"Number of lags for endogenous variables: {self.args['lags'][0]}")
        print(f"Number of lags for weakly exogenous variables: {self.args['lags'][1]}")
        print(f"Number of posterior draws: {self.args['draws']}/{self.args['thin']}={self.args['draws']//self.args['thin']}")
        if self.args.get('eigen', False):
            print(f"Number of stable posterior draws: {len(self.stacked_results.get('F_eigen', []))}")
        print(f"Number of cross-sectional units: {len(self.gW)}")
        print("-" * 75)
        print("Convergence diagnostics")
        print(f"Geweke statistic: {CD['perc']}")
        print("-" * 75)
        print("F-test, first order serial autocorrelation of cross-unit residuals")
        print("Summary statistics:")
        print(pd.DataFrame(res['p.res'], 
                          columns=['Count', 'Percentage'],
                          index=['>0.10', '0.05-0.10', '0.01-0.05', '<=0.01']))
        print("-" * 75)
        print("Average pairwise cross-unit correlation of unit-model residuals")
        print(f"Data correlations: {cross_corr.get('data.cor', {})}")
        print(f"Residual correlations: {cross_corr.get('resid.cor', {})}")
        print("-" * 75)
        
        return summary_dict
    
    def residuals(self):
        """
        Calculate residuals of the global and country models.
        
        Returns
        -------
        dict
            Dictionary containing 'global' and 'country' residuals arrays.
        """
        if self.args.get('thindraws', 0) == 0:
            print("Computation of BGVAR has yielded no stable posterior draws!")
            return None
        
        G_mat = self.stacked_results['Ginv_large']
        A_mat = self.stacked_results['A_large']
        lags = self.args['lags']
        pmax = max(lags)
        draws = self.args.get('thindraws', A_mat.shape[2])
        time = self.args['time']
        trend = self.args['trend']
        xglobal = self.xglobal
        
        YY = xglobal.iloc[pmax:].values
        XX_lag = helpers.mlag(xglobal.values, pmax)
        XX = np.hstack([XX_lag[pmax:], np.ones((YY.shape[0], 1))])
        if trend:
            XX = np.hstack([XX, np.arange(1, YY.shape[0] + 1).reshape(-1, 1)])
        
        time_labels = time.iloc[pmax:] if isinstance(time, pd.Series) else time[pmax:]
        
        res_array_country = np.zeros((draws, *YY.shape))
        res_array_global = np.zeros((draws, *YY.shape))
        
        for irep in range(draws):
            res_array_global[irep] = YY - XX @ A_mat[:, :, irep].T
            res_array_country[irep] = res_array_global[irep] @ np.linalg.inv(G_mat[:, :, irep]).T
        
        return {
            'global': res_array_global,
            'country': res_array_country,
            'Data': pd.DataFrame(YY, index=time_labels, columns=xglobal.columns)
        }
    
    def coef(self, quantile=0.50):
        """
        Extract model coefficients at specified quantiles.
        
        Parameters
        ----------
        quantile : float or array-like, default=0.50
            Quantile(s) to extract (between 0 and 1).
        
        Returns
        -------
        np.ndarray
            Array of coefficients with shape (K, K, p) or (q, K, K, p) if multiple quantiles.
        """
        A_large = self.stacked_results['A_large']
        lags = self.args['lags']
        pmax = max(lags)
        K = self.xglobal.shape[1]
        
        if isinstance(quantile, (list, tuple, np.ndarray)):
            quantiles = np.array(quantile)
            coef_array = np.zeros((len(quantiles), K, K * pmax + 1 + int(self.args['trend'])))
            for i, q in enumerate(quantiles):
                coef_array[i] = np.quantile(A_large, q, axis=2)
            return coef_array
        else:
            return np.quantile(A_large, quantile, axis=2)
    
    def vcov(self, quantile=0.50):
        """
        Extract variance-covariance matrix at specified quantile.
        
        Parameters
        ----------
        quantile : float, default=0.50
            Quantile to extract (between 0 and 1).
        
        Returns
        -------
        np.ndarray
            Variance-covariance matrix (K x K).
        """
        S_large = self.stacked_results['S_large']
        return np.quantile(S_large, quantile, axis=2)
    
    def fitted(self, global_model=True):
        """
        Extract fitted values.
        
        Parameters
        ----------
        global_model : bool, default=True
            If True, returns global model fitted values. Otherwise country model fitted values.
        
        Returns
        -------
        pd.DataFrame
            Fitted values.
        """
        residuals_dict = self.residuals()
        YY = residuals_dict['Data']
        
        if global_model:
            res = np.median(residuals_dict['global'], axis=0)
        else:
            res = np.median(residuals_dict['country'], axis=0)
        
        fitted_vals = YY.values - res
        return pd.DataFrame(fitted_vals, index=YY.index, columns=YY.columns)
    
    def logLik(self, quantile=0.50):
        """
        Calculate log-likelihood at specified quantile.
        
        Parameters
        ----------
        quantile : float, default=0.50
            Quantile to use for calculation.
        
        Returns
        -------
        float
            Log-likelihood value.
        """
        from scipy.stats import multivariate_normal
        
        A_mat = np.quantile(self.stacked_results['A_large'], quantile, axis=2)
        S_mat = np.quantile(self.stacked_results['S_large'], quantile, axis=2)
        
        lags = self.args['lags']
        pmax = max(lags)
        xglobal = self.xglobal
        
        YY = xglobal.iloc[pmax:].values
        XX_lag = helpers.mlag(xglobal.values, pmax)
        XX = np.hstack([XX_lag[pmax:], np.ones((YY.shape[0], 1))])
        if self.args['trend']:
            XX = np.hstack([XX, np.arange(1, YY.shape[0] + 1).reshape(-1, 1)])
        
        residuals = YY - XX @ A_mat.T
        
        try:
            loglik = multivariate_normal.logpdf(residuals, mean=np.zeros(YY.shape[1]), cov=S_mat)
            return np.sum(loglik)
        except:
            return -np.inf
    
    def dic(self):
        """
        Calculate Deviance Information Criterion (DIC).
        
        Returns
        -------
        dict
            Dictionary containing DIC, effective number of parameters (pD), and deviance.
        """
        from scipy.stats import multivariate_normal
        
        A_large = self.stacked_results['A_large']
        S_large = self.stacked_results['S_large']
        draws = A_large.shape[2]
        
        lags = self.args['lags']
        pmax = max(lags)
        xglobal = self.xglobal
        
        YY = xglobal.iloc[pmax:].values
        XX_lag = helpers.mlag(xglobal.values, pmax)
        XX = np.hstack([XX_lag[pmax:], np.ones((YY.shape[0], 1))])
        if self.args['trend']:
            XX = np.hstack([XX, np.arange(1, YY.shape[0] + 1).reshape(-1, 1)])
        
        # Calculate deviance for each draw
        deviances = np.zeros(draws)
        for irep in range(draws):
            residuals = YY - XX @ A_large[:, :, irep].T
            try:
                loglik = multivariate_normal.logpdf(residuals, mean=np.zeros(YY.shape[1]), cov=S_large[:, :, irep])
                deviances[irep] = -2 * np.sum(loglik)
            except:
                deviances[irep] = np.inf
        
        # Mean deviance
        D_bar = np.mean(deviances[np.isfinite(deviances)])
        
        # Deviance at posterior mean
        A_mean = np.mean(A_large, axis=2)
        S_mean = np.mean(S_large, axis=2)
        residuals_mean = YY - XX @ A_mean.T
        try:
            loglik_mean = multivariate_normal.logpdf(residuals_mean, mean=np.zeros(YY.shape[1]), cov=S_mean)
            D_theta_bar = -2 * np.sum(loglik_mean)
        except:
            D_theta_bar = np.inf
        
        # Effective number of parameters
        pD = D_bar - D_theta_bar
        
        # DIC
        DIC = D_bar + pD
        
        return {
            'DIC': DIC,
            'pD': pD,
            'D_bar': D_bar,
            'D_theta_bar': D_theta_bar
        }
    
    def __repr__(self):
        """String representation of the model."""
        prior_names = {
            'MN': 'Minnesota prior (MN)',
            'SSVS': 'SSVS prior (SSVS)',
            'NG': 'Normal-Gamma prior (NG)',
            'HS': 'Horseshoe prior (HS)'
        }
        
        return (f"BGVAR Model\n"
                f"Prior: {prior_names.get(self.args['prior'], self.args['prior'])}\n"
                f"Lags: {self.args['lags'][0]} (endo), {self.args['lags'][1]} (w. exog)\n"
                f"Countries: {self.N}\n"
                f"Variables: {self.xglobal.shape[1]}\n"
                f"Draws: {self.args['draws']}/{self.args['thin']} = {self.args['draws']//self.args['thin']}")


# Prior modules will be implemented separately in prior.py

