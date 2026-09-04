import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

data = np.load("results/duffing_oscilliator.npz")

t = data["t"]
x = data["x"]
v = data["v"]

X = np.column_stack((x, v))
dt = t[1] - t[0]

# Create a polynomial library
library = ps.PolynomialLibrary(degree=3, include_bias=True)

# Use sequential thresholded least squares to retain only important terms
optimizer = ps.STLSQ(threshold=0.05, alpha=1e-6)

# Create the SINDy model using the library and sparse optimizer
model = ps.SINDy(feature_library=library, optimizer=optimizer)

# Fit the model to displacement and velocity data with time step dt
model.fit(X, t=dt, feature_names=["x", "v"])

print("\nDiscovered model:")
model.print()

# Define an unseen initial condition for validation
initial_state_test = [0.4, 1.0]

# Solve the original Duffing equation for the test case
def duffing_rhs(t, state):
    x, v = state
    return [v, -0.30 * v - 1.00 * x - 0.50 * x**3]

reference = solve_ivp(
    duffing_rhs,
    (t[0], t[-1]),
    initial_state_test,
    t_eval=t
)

# Simulate the equation discovered by SINDy
sindy_prediction = model.simulate(initial_state_test, t)

# Plot displacement from both models
plt.figure(figsize=(8, 4))
plt.plot(t, reference.y[0], label="Original Duffing solver")
plt.plot(t, sindy_prediction[:, 0], "--", label="SINDy model")
plt.xlabel("Time")
plt.ylabel("Displacement x")
plt.title("Validation on an unseen initial condition")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("results/sindy_validation.png", dpi=150)
plt.show()


# Compute relative error between reference and SINDy trajectories (normalized)
relative_error = np.linalg.norm(reference.y.T - sindy_prediction) / np.linalg.norm(reference.y.T)

print(f"Relative validation error: {relative_error:.2e}")