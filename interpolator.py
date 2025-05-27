import numpy as np


def get_bilinear_indices_and_weights(P_target, T_target, P_grid, T_grid):
    """
    Given a target pressure and temperature find the nearest pressures and temperatures on the given grid above and below

    o------o -> (Pressure)
    |      |
    |    x |
    o------o
    |
    V
    (Temperature)       Do you like my ASCII art? Yes, a human wrote this!

    We will find the indices corresponding to each of the four surrounding points in the grid (marked as 'o's in my picture) 
    and return their indices and along with weights for bilinear interpolation.

    We treat pressure logarithmically and temperature linearly.
    """
    # Ensure P_target and T_target are within the grid bounds
    assert P_target >= P_grid.min() and P_target <= P_grid.max(), f"Pressure {P_target} is out of bounds for the grid."
    assert T_target >= T_grid.min() and T_target <= T_grid.max(), f"Temperature {T_target} is out of bounds for the grid."

    # Convert pressure to log10 scale
    log_P_grid = np.log10(P_grid)
    log_P_target = np.log10(P_target)

    # Find index *below* for pressure
    i_p = np.searchsorted(log_P_grid, log_P_target) - 1
    i_p = np.clip(i_p, 0, len(P_grid) - 2)  # avoid out-of-bounds

    # Find index *below* for temperature
    i_t = np.searchsorted(T_grid, T_target) - 1
    i_t = np.clip(i_t, 0, len(T_grid) - 2)

    # Grid values at surrounding points
    P0, P1 = log_P_grid[i_p], log_P_grid[i_p + 1]
    T0, T1 = T_grid[i_t], T_grid[i_t + 1]

    # Linear interpolation weights
    wp1 = (log_P_target - P0) / (P1 - P0)  # weight for i_p+1
    wp0 = 1 - wp1                          # weight for i_p

    wt1 = (T_target - T0) / (T1 - T0)      # weight for i_t+1
    wt0 = 1 - wt1                          # weight for i_t

    # Final weights for 2D bilinear interpolation
    weights = np.array([
        wt0 * wp0,  # (i_p,   i_t)
        wt0 * wp1,  # (i_p+1, i_t)
        wt1 * wp0,  # (i_p,   i_t+1)
        wt1 * wp1   # (i_p+1, i_t+1)
    ])

    # Corresponding index pairs
    indices = [
        (i_p,   i_t),
        (i_p+1, i_t),
        (i_p,   i_t+1),
        (i_p+1, i_t+1)
    ]

    return indices, weights


def get_interpolated_cross_sections(pressure, temperature, absorber):
    """
    Get the interpolated cross section for a given pressure and temperature.
    """
    # Get the indices and weights for bilinear interpolation
    indices, weights = get_bilinear_indices_and_weights(pressure, temperature, absorber.pressures, absorber.temperatures)

    # Extract the cross sections at the surrounding grid points
    xsecs = np.array([absorber.cross_sections[i_p, i_t] for (i_p, i_t) in indices])

    # Replace any 0's with 1e-99 to avoid log10(0)
    xsecs[xsecs == 0] = 1e-99

    # Perform bilinear interpolation. We do this on the "logged" cross sections!
    interpolated_xsec = np.power(10, np.sum(np.log10(xsecs) * weights[:, np.newaxis], axis=0))

    # print(f"Interpolated cross section for Absorber {absorber.molecule_name} P={pressure} atm, T={temperature} K: {interpolated_xsec}")
    # # In CO some of the cross sections are zero so this gives nans. We should replace them with 0.
    # interpolated_xsec[np.isnan(interpolated_xsec)] = 0

    return interpolated_xsec