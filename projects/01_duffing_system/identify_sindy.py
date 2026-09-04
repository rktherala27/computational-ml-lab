import numpy as np
import pysindy as ps

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