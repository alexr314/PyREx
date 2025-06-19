"""
Visualization utilities for Bayesian retrieval results.

Functions for creating corner plots, trace plots, and other diagnostic visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import corner


def plot_corner(posterior_samples, gas_names=None, true_values=None, **corner_kwargs):
    """
    Create corner plot of posterior distributions.
    
    Parameters
    ----------
    posterior_samples : array_like
        Posterior samples of shape (nsamples, ndim)
    gas_names : list, optional
        List of parameter names for labels
    true_values : array_like, optional
        True parameter values to mark on plot
    **corner_kwargs : dict
        Additional arguments passed to corner.corner()
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Corner plot figure
    """
    # Default gas names
    if gas_names is None:
        gas_names = ['CO2', 'CO', 'NH3', 'CH4', 'H2O']
    
    # Default corner plot settings
    default_kwargs = {
        'labels': gas_names,
        'show_titles': True,
        'title_kwargs': {'fontsize': 12},
        'label_kwargs': {'fontsize': 14},
        'hist_kwargs': {'density': True},
        'plot_contours': True,
        'fill_contours': True,
        'levels': [0.68, 0.95],
        'color': 'blue',
        'alpha': 0.6
    }
    default_kwargs.update(corner_kwargs)
    
    # Create corner plot
    fig = corner.corner(posterior_samples, **default_kwargs)
    
    # Mark true values if provided
    if true_values is not None:
        corner.overplot_lines(fig, true_values, color='red', linewidth=2, linestyle='--')
        corner.overplot_points(fig, true_values[None], marker='o', color='red', markersize=8)
    
    return fig


def plot_spectrum_fit(wavelengths, y_obs, posterior_samples, fixed_params, gas_names=None, 
                     sigma=None, true_spectrum=None, n_posterior_draws=100):
    """
    Plot observed spectrum with posterior prediction bands.
    
    Parameters
    ----------
    wavelengths : array_like
        Wavelength array
    y_obs : array_like
        Observed spectrum
    posterior_samples : array_like
        Posterior samples of gas concentrations
    fixed_params : dict
        Fixed model parameters
    gas_names : list, optional
        Gas names corresponding to posterior samples
    sigma : float, optional
        Observation noise level for error bars
    true_spectrum : array_like, optional
        True spectrum (for synthetic data)
    n_posterior_draws : int
        Number of posterior samples to draw for prediction band
        
    Returns
    -------
    fig, ax : matplotlib objects
        Figure and axis objects
    """
    from forward_model import compute_binned_modulations
    
    if gas_names is None:
        gas_names = ['CO2', 'CO', 'NH3', 'CH4', 'H2O']
    
    # Draw random posterior samples
    n_samples = len(posterior_samples)
    draw_indices = np.random.choice(n_samples, size=min(n_posterior_draws, n_samples), replace=False)
    
    # Compute model predictions for posterior draws
    predictions = []
    for idx in draw_indices:
        conc_dict = {name: conc for name, conc in zip(gas_names, posterior_samples[idx])}
        pred = compute_binned_modulations(concentrations=conc_dict, **fixed_params)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Compute prediction statistics
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)
    pred_lower = np.percentile(predictions, 16, axis=0)
    pred_upper = np.percentile(predictions, 84, axis=0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot observations
    if sigma is not None:
        ax.errorbar(wavelengths, y_obs, yerr=sigma, fmt='o', color='black', 
                   alpha=0.7, markersize=4, label='Observations')
    else:
        ax.plot(wavelengths, y_obs, 'ko', alpha=0.7, markersize=4, label='Observations')
    
    # Plot true spectrum if provided
    if true_spectrum is not None:
        ax.plot(wavelengths, true_spectrum, 'r-', linewidth=2, label='True spectrum')
    
    # Plot posterior mean and uncertainty
    ax.plot(wavelengths, pred_mean, 'b-', linewidth=2, label='Posterior mean')
    ax.fill_between(wavelengths, pred_lower, pred_upper, alpha=0.3, color='blue', 
                   label='68% confidence')
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Modulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Spectrum Fit')
    
    return fig, ax


def plot_parameter_traces(posterior_samples, gas_names=None):
    """
    Plot parameter traces (useful for checking convergence).
    
    Parameters
    ----------
    posterior_samples : array_like
        Posterior samples of shape (nsamples, ndim)
    gas_names : list, optional
        Parameter names for labels
        
    Returns
    -------
    fig, axes : matplotlib objects
        Figure and axes objects
    """
    if gas_names is None:
        gas_names = ['CO2', 'CO', 'NH3', 'CH4', 'H2O']
    
    ndim = posterior_samples.shape[1]
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2*ndim), sharex=True)
    
    if ndim == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, gas_names)):
        ax.plot(posterior_samples[:, i], alpha=0.7)
        ax.set_ylabel(f'{name}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample number')
    plt.tight_layout()
    
    return fig, axes


def plot_posterior_summary(posterior_samples, gas_names=None, true_values=None):
    """
    Create summary plot with marginal distributions and statistics.
    
    Parameters
    ----------
    posterior_samples : array_like
        Posterior samples of shape (nsamples, ndim)
    gas_names : list, optional
        Parameter names
    true_values : array_like, optional
        True parameter values
        
    Returns
    -------
    fig, axes : matplotlib objects
        Figure and axes objects
    """
    if gas_names is None:
        gas_names = ['CO2', 'CO', 'NH3', 'CH4', 'H2O']
    
    ndim = posterior_samples.shape[1]
    fig, axes = plt.subplots(1, ndim, figsize=(3*ndim, 4))
    
    if ndim == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, gas_names)):
        # Plot histogram
        ax.hist(posterior_samples[:, i], bins=50, density=True, alpha=0.7, color='skyblue')
        
        # Compute statistics
        median = np.median(posterior_samples[:, i])
        q16, q84 = np.percentile(posterior_samples[:, i], [16, 84])
        
        # Mark statistics
        ax.axvline(median, color='blue', linewidth=2, label=f'Median: {median:.2e}')
        ax.axvline(q16, color='blue', linewidth=1, linestyle='--', alpha=0.7)
        ax.axvline(q84, color='blue', linewidth=1, linestyle='--', alpha=0.7)
        
        # Mark true value if provided
        if true_values is not None:
            ax.axvline(true_values[i], color='red', linewidth=2, linestyle=':', 
                      label=f'True: {true_values[i]:.2e}')
        
        ax.set_xlabel(f'{name} concentration')
        ax.set_ylabel('Posterior density')
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes
