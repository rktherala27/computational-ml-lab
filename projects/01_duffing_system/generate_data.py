import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

c = 0.30
k = 1.00
alpha = 0.50

def duffing_rhs(t, state):
    x, v = state
    dxdt = v
    dvdt = -c * v - k * x - alpha * x**3
    return [dxdt, dvdt]

t = np.arange(0.0, 20.0, 0.01)
initial_state = [1.4, 0.0]

solution = solve_ivp(
    duffing_rhs,
    t_span=(t[0], t[-1]),
    y0=initial_state,
    t_eval=t,
    rtol=1e-9,
    atol=1e-11,
)

x = solution.y[0]
v = solution.y[1]

np.savez("results/duffing_oscilliator.npz", t=t, x=x, v=v)

plt.figure(figsize=(8, 4))
plt.plot(t, x)
plt.xlabel("Time")
plt.ylabel("Displacement x")
plt.title("Damped Duffing oscillator")
plt.grid()
plt.tight_layout()
plt.savefig("results/displacement_plot.png", dpi=150)
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(x, v)
plt.xlabel("Displacement x")
plt.ylabel("Velocity v")
plt.title("Phase plot")
plt.grid()
plt.tight_layout()
plt.savefig("results/phase_plot.png", dpi=150)
plt.show()