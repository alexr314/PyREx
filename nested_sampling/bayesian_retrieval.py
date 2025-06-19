"""
Bayesian retrieval pipeline for exoplanet transit spectra using nested sampling.

This module implements the core functionality for Bayesian parameter estimation
of atmospheric gas concentrations using the forward model and dynesty nested sampling.
"""

import numpy as np
import dynesty
from dynesty import utils as dyfunc
import sys
import os

# Add parent directory to path to import PyREx modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forward_model import compute_binned_modulations


class ExoplanetRetrieval:
    """
    Bayesian retrieval class for exoplanet atmospheric parameters.
    
    Parameters
    ----------
    fixed_params : dict
        Dictionary containing fixed planetary and stellar parameters:
        - temperature : float
        - pressure : float  
        - planet_radius : float
        - star_radius : float
        - g : float (gravitational acceleration)
        
    gas_names : list
        List of gas names in order (default: ['CO2', 'CO', 'NH3', 'CH4', 'H2O'])
        
    prior_bounds : tuple
        (min, max) concentration bounds for log-uniform priors (default: (1e-8, 1))
        
    sigma : float
        Gaussian noise standard deviation for likelihood (default: 10e-6)
    """
    
    def __init__(self, fixed_params, gas_names=None, prior_bounds=(1e-8, 1), sigma=10e-6):
        self.fixed_params = fixed_params
        self.gas_names = gas_names or ['CO2', 'CO', 'NH3', 'CH4', 'H2O']
        self.ndim = len(self.gas_names)
        self.prior_bounds = prior_bounds
        self.sigma = sigma
        
        # Precompute log bounds for efficiency
        self.log_min = np.log10(prior_bounds[0])
        self.log_max = np.log10(prior_bounds[1])
        
        # Observations will be set later
        self.y_obs = None
        
    def prior_transform(self, u):
        """
        Transform unit cube samples to physical parameters using log-uniform priors.
        
        Parameters
        ----------
        u : array_like
            Unit cube samples of shape (ndim,)
            
        Returns
        -------
        concentrations : array_like
            Physical gas concentrations
        """
        log_concentrations = self.log_min + u * (self.log_max - self.log_min)
        return 10**log_concentrations
    
    def log_likelihood(self, concentrations):
        """
        Compute log-likelihood assuming Gaussian noise per spectral bin.
        
        Parameters
        ----------
        concentrations : array_like
            Gas concentrations in order of self.gas_names
            
        Returns
        -------
        loglike : float
            Log-likelihood value
        """
        # Create concentration dictionary
        conc_dict = {name: conc for name, conc in zip(self.gas_names, concentrations)}
        
        # Compute forward model prediction
        predicted = compute_binned_modulations(
            concentrations=conc_dict,
            **self.fixed_params
        )
        
        # Compute log-likelihood (assuming independent Gaussian noise)
        chi2 = np.sum(((self.y_obs - predicted) / self.sigma)**2)
        return -0.5 * chi2
    
    def generate_synthetic_data(self, true_concentrations, noise_seed=42):
        """
        Generate synthetic observations from known concentrations plus noise.
        
        Parameters
        ----------
        true_concentrations : array_like or dict
            True gas concentrations. If array, assumed to be in order of self.gas_names
        noise_seed : int
            Random seed for noise generation
            
        Returns
        -------
        y_obs : array_like
            Synthetic observations with noise
        true_spectrum : array_like  
            True spectrum without noise
        """
        np.random.seed(noise_seed)
        
        # Handle both dict and array inputs
        if isinstance(true_concentrations, dict):
            true_conc_dict = true_concentrations
        else:
            true_conc_dict = {name: conc for name, conc in zip(self.gas_names, true_concentrations)}
        
        # Generate true spectrum
        true_spectrum = compute_binned_modulations(
            concentrations=true_conc_dict,
            **self.fixed_params
        )
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.sigma, len(true_spectrum))
        y_obs = true_spectrum + noise
        
        # Store observations
        self.y_obs = y_obs
        
        return y_obs, true_spectrum
    
    def run_nested_sampling(self, nlive=500, dlogz=0.1, **kwargs):
        """
        Run dynesty nested sampling to estimate posterior.
        
        Parameters
        ----------
        nlive : int
            Number of live points
        dlogz : float
            Target evidence uncertainty for stopping criterion
        **kwargs : dict
            Additional arguments passed to dynesty.NestedSampler
            
        Returns
        -------
        results : dynesty.results.Results
            Sampling results containing samples, weights, evidence, etc.
        """
        if self.y_obs is None:
            raise ValueError("No observations set. Call generate_synthetic_data() first or set self.y_obs manually.")
        
        # Initialize sampler
        sampler = dynesty.NestedSampler(
            self.log_likelihood, 
            self.prior_transform, 
            ndim=self.ndim,
            nlive=nlive,
            **kwargs
        )
        
        # Run sampling
        sampler.run_nested(dlogz=dlogz)
        
        return sampler.results
    
    def process_results(self, results):
        """
        Process dynesty results to extract posterior samples.
        
        Parameters
        ----------
        results : dynesty.results.Results
            Results from run_nested_sampling()
            
        Returns
        -------
        posterior_samples : array_like
            Equal-weight posterior samples of shape (nsamples, ndim)
        evidence : float
            Log evidence estimate
        evidence_err : float
            Log evidence uncertainty
        """
        # Extract samples and weights
        samples = results.samples
        weights = np.exp(results.logwt - results.logz[-1])
        
        # Resample to get equal-weight posterior samples
        posterior_samples = dyfunc.resample_equal(samples, weights)
        
        # Extract evidence
        evidence = results.logz[-1]
        evidence_err = results.logzerr[-1]
        
        return posterior_samples, evidence, evidence_err


def create_fixed_params():
    """
    Create default fixed parameters using PyREx constants.
    
    Returns
    -------
    fixed_params : dict
        Dictionary of fixed planetary/stellar parameters
    """
    # Import constants
    from physical_constants import RJ, Rsun, MJ, Gnewton
    
    # Set parameters matching forward_model_demo2.ipynb
    temperature = 300  # K
    pressure = 10      # bars
    planet_radius = RJ  # m
    star_radius = Rsun  # m  
    planet_mass = MJ    # kg
    g = Gnewton * planet_mass / (planet_radius**2)  # m/s^2
    
    return {
        'temperature': temperature,
        'pressure': pressure,
        'planet_radius': planet_radius,
        'star_radius': star_radius,
        'g': g
    }
