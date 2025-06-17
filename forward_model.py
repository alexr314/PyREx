import numpy as np
import matplotlib.pyplot as plt
import os
from utils import LineList
from scipy.special import exp1
from physical_constants import RJ, Rsun, MJ, k_B, Gnewton, mamu, gamma
from interpolator import get_interpolated_cross_sections
from ariel_wavelengths import ariel_wavelengths, ariel_bins


opacity_data_path = '../line_lists'

# get all files in the directory which end with .h5
line_list_paths = [f for f in os.listdir(opacity_data_path) if f.endswith('.h5')]

absorbers = [LineList(os.path.join(opacity_data_path, f)) for f in line_list_paths]

# Assert that all absorbers have the same wavelengths
for i in range(len(absorbers)-1):
    assert np.array_equal(absorbers[i].wavelengths, absorbers[i+1].wavelengths), \
        f"Wavelengths of {absorbers[i].name} and {absorbers[i+1].name} do not match."
        
# Since they are the same, get the wavelengths from the first absorber
wavelengths = absorbers[0].wavelengths


# Assert that all the temperature grids and pressure grids are the same
for i in range(len(absorbers)-1):
    assert np.array_equal(absorbers[i].pressures, absorbers[i+1].pressures)
    assert np.array_equal(absorbers[i].temperatures, absorbers[i+1].temperatures)

# Since they are the same, get the pressure and temperature grids from the first absorber
temp_grid = absorbers[0].temperatures
pressure_grid = absorbers[0].pressures

# print('Temperature grid:', temp_grid)
# print('Pressure grid:', pressure_grid)


def get_m_bar(concentrations):
    """
    Calculate the mean molecular mass of the atmosphere.
    concentrations: list of absolute concentrations of each absorber

    We include the contribution of H2 and He in the mean molecular mass, 
    which are assumed to have a relative concentration of 0.17 which gives a mean molecular mass of 2.34

    Returns the mean molecular mass in amu.
    """

    # Start with the contribution of H2 and He
    mbar = (1 - np.sum(list(concentrations.values()))) * 2.34

    # Add the contributions to the mass from each absorber
    for absorber in absorbers:
        mbar += concentrations[absorber.molecule_name] * absorber.molecular_mass

    return mbar


def get_kappa(concentrations, temperature, pressure):
    """
    Computes the mass absorption coefficient (kappa) in m²/kg at a given pressure and temperature.

    Parameters:
        concentrations: dict[str, float]
            Absolute concentrations of each absorber, keyed by molecule name.
        temperature: float
            Atmospheric temperature in Kelvin.
        pressure: float
            Atmospheric pressure in bar.

    Returns:
        kappa: np.ndarray
            1D array of opacities [m²/kg] at each wavelength.
    """

    # Compute mean molecular mass
    m_bar = get_m_bar(concentrations)  # amu
    m_bar_kg = m_bar * mamu            # kg

    # Initialize opacity array
    kappa = np.zeros(len(wavelengths))

    # Loop over absorbers
    for absorber in absorbers:
        # Get the name and make sure it exists in the concentrations dict
        name = absorber.molecule_name
        if name not in concentrations:
            raise ValueError(f"Missing concentration for absorber: '{name}'")

        ### We will implement this formula: X_a * m_a * chi_a / m_bar

        concentration = concentrations[name] # This is $X_a$, the absolute concentration of the absorber

        # Get the molecular mass of the absorber and convert it to kg
        molecular_mass_kg = absorber.molecular_mass * mamu # This is $m_a$

        # Get the interpolated cross sections for the absorber at the given pressure and temperature
        chi = get_interpolated_cross_sections(pressure, temperature, absorber) # This is $\Chi_a$ in cm²/molecule
        chi_meters = chi * 1e-4 # in m²/molecule
        chi_SI = chi_meters / molecular_mass_kg # in m²/kg # Allegedly this is the correct way to do it???

        kappa += concentration * molecular_mass_kg * chi_SI / m_bar_kg

    return kappa


def compute_scale_height(T, g, concentrations):
    """
    Compute the scale height of the atmosphere, given:
    - T: temperature in Kelvin
    - g: gravitational acceleration in m/s^2
    - concentrations: relative abundances of the species in the atmosphere (these are used to compute the mean molecular weight)
    
    Retrurns the scale height in meters!
    """
    m_bar = get_m_bar(concentrations)  # mean molecular weight in amu
    m = m_bar * mamu  # convert mean molecular weight to kg
    return (k_B * T) / (m * g)


def get_tau(concentrations, temperature, pressure, planet_radius, g):
    """
    Calculate the optical depth of the atmosphere.
    """
    # kappa = get_kappa(concentrations, temperature, pressure) # in cm²/molecule
    kappa = get_kappa(concentrations, temperature, pressure) # in m²/kg
    H = compute_scale_height(temperature, g, concentrations) # In meters
    pressure_SI = pressure * 1e5  # Convert pressure from atm to Pa
    tau = (pressure_SI * kappa / g) * np.sqrt(2 * np.pi * planet_radius / H)
    return tau   # Which is dimensionless!


def compute_scale_height(T, g, concentrations):
    """
    Compute the scale height of the atmosphere, given:
    - T: temperature in Kelvin
    - g: gravitational acceleration in m/s^2
    - concentrations: relative abundances of the species in the atmosphere (these are used to compute the mean molecular weight)
    
    Retrurns the scale height in meters!
    """
    m_bar = get_m_bar(concentrations)  # mean molecular weight in amu
    m = m_bar * mamu  # convert mean molecular weight to kg
    return (k_B * T) / (m * g)


def get_tau(concentrations, temperature, pressure, planet_radius, g):
    """
    Calculate the optical depth of the atmosphere.
    """
    # kappa = get_kappa(concentrations, temperature, pressure) # in cm²/molecule
    kappa = get_kappa(concentrations, temperature, pressure) # in m²/kg
    H = compute_scale_height(temperature, g, concentrations) # In meters
    pressure_SI = pressure * 1e5  # Convert pressure from atm to Pa
    tau = (pressure_SI * kappa / g) * np.sqrt(2 * np.pi * planet_radius / H)
    return tau   # Which is dimensionless!


def compute_modulation(concentrations, temperature, pressure, planet_radius, star_radius, g):
    """
    Compute the modulation of the light curve due to the atmosphere.
    """
    tau = get_tau(concentrations, temperature, pressure, planet_radius, g)
    H = compute_scale_height(temperature, g, concentrations)
    first_term = planet_radius / star_radius
    second_term = (H / star_radius) * (gamma + np.log(tau) + exp1(tau))
    return first_term + second_term


def bin_modulations(modulations, bins=ariel_bins):
    """
    Bin the modulations into the specified bins.
    """
    binned_modulations = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (bins[i] <= wavelengths) & (wavelengths < bins[i + 1])
        binned_modulations[i] = np.mean(modulations[mask])
    return binned_modulations



# # This is an alternative way to bin the modulations using a binning matrix
# # It is 10x faster than the above method, but requires more memory
# # We can switch to this if we determine that this step is a bottleneck in the code.

# binning_matrix = np.empty((len(ariel_bins) - 1, len(wavelengths)))

# for i in range(len(ariel_bins) - 1):
#     mask = (wavelengths >= ariel_bins[i]) & (wavelengths < ariel_bins[i + 1])
#     if mask.sum() == 0:
#         raise ValueError(f"No wavelengths found in bin {i} ({ariel_bins[i]} - {ariel_bins[i + 1]})")
#     vec = mask.astype(float) / mask.sum()
#     binning_matrix[i, :] = vec

# def bin_modulations_with_matrix(modulations, binning_matrix):
#     """
#     Bin the modulations using the provided binning matrix.
#     """
#     binned_modulations = np.dot(binning_matrix, modulations)
#     return binned_modulations


def compute_binned_modulations(concentrations, temperature, pressure, planet_radius, star_radius, g, bins=ariel_bins):
    """
    Compute the binned modulations of the light curve.
    """
    modulations = compute_modulation(concentrations, temperature, pressure, planet_radius, star_radius, g)
    binned_modulations = bin_modulations(modulations, bins)
    return binned_modulations