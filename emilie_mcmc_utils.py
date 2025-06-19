import numpy as np

# random choices within those bounds  
# This provides the starting point for the MCMC chain

def random_choice(temp_min, temp_max, plrad_min, plrad_max, starrad_min, starrad_max):
    # chosing three radnom values for our fitted parameters
    temp_rand = np.random.uniform(temp_min,temp_max) 
    plrad_rand = np.random.uniform(plrad_min,plrad_max)
    starrad_rand = np.random.uniform(starrad_min, starrad_max)
    return np.array([temp_rand, plrad_rand, starrad_rand])


# function to calculate the log likelihood
def get_log_likelihood(model, obs, sigma):  
    chi2 = np.sum(((obs - model) / sigma) ** 2)
    return -0.5 * chi2


def sample_proposal(current_state, sigmas):
    """
    Given the current state, sample a new state from a multivariate Gaussian with the given standard deviations.
    """
    return np.random.normal(current_state, sigmas)