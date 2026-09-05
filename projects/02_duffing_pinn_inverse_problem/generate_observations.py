import numpy as np
from scipy.integrate import solve_ivp

# True physical parameters used only to generate reference data
c_true = 0.30
k = 1.00
alpha = 0.50

def duffing_rhs(t, state):
    x, v = state
    return [v, -c_true * v - k * x - alpha * x**3]

# Dense reference trajectory
t_full = np.linspace(0.0, 20.0, 2001)

solution = solve_ivp(duffing_rhs, (t_full[0], t_full[-1]), [1.4, 0.0],
    t_eval=t_full, rtol=1e-9, atol=1e-11)

x_full = solution.y[0]

# Keep only 25 noisy displacement observations
np.random.seed(42)
observation_indices = np.linspace(0, len(t_full) - 1, 25, dtype=int)

t_obs = t_full[observation_indices]
x_obs_clean = x_full[observation_indices]

noise_std = 0.02
x_obs = x_obs_clean + noise_std * np.random.randn(len(x_obs_clean))

np.savez("results/duffing_observations.npz", t_full=t_full, x_full=x_full,
    t_obs=t_obs, x_obs=x_obs, c_true=c_true)

# print("Saved 25 noisy displacement observations.")
print(f"True damping coefficient: c = {c_true}")