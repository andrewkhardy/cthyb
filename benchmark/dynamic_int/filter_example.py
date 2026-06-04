import numpy as np

def smooth_and_shift_Q(Q_l_data, static_shift, n_leg, beta, n_tau):
    from triqs.gfs import GfImTime
    from scipy.special import eval_legendre
    
    tau_mesh = np.linspace(0, beta, n_tau)
    Q_tau_smooth = np.zeros(n_tau)
    
    # Gaussian filter to suppress high-frequency Monte Carlo noise
    # Adjust sigma based on where the noise floor begins
    sigma = 15.0 
    
    for n in range(n_leg):
        filter_factor = np.exp(-0.5 * (n / sigma)**2)
        q_n = Q_l_data[n].real
        
        # Evaluate Legendre polynomial
        x = 2.0 * tau_mesh / beta - 1.0
        P_n = eval_legendre(n, x)
        
        Q_tau_smooth += filter_factor * q_n * P_n
        
    # Add the missing static shift (integration constant)
    return Q_tau_smooth + static_shift

