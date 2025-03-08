import numpy as np
from scipy.stats import gaussian_kde

def joint_eti_region(sample, alpha):
    # Compute the marginal quantiles for each parameter
    quantiles = np.quantile(sample, [alpha/4, 1-alpha/4], axis=0)
                
    # Create the joint ETI region
    eti_region = np.array([[quantiles[0, 0], quantiles[0, 1]],
                            [quantiles[1, 0], quantiles[1, 1]]])
                            
    return eti_region
