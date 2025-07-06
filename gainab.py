import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
L = 300  # total cavity length in µm
dz = 1   # spatial step in µm
Z = np.arange(0, L, dz)
Nz = len(Z)

# Gain section: first 200 µm, Absorber: last 100 µm
gain_mask = Z < 200
abs_mask = Z >= 200

# Initial pulse (Gaussian)
z0 = 150
sigma = 10
E = np.exp(-((Z - z0) ** 2) / (2 * sigma ** 2)).astype(np.complex128)

# Initial carrier densities
N_g = np.full(Nz, 1.5e24)  # gain section carrier density [1/m^3]
N_a = np.full(Nz, 1.0e24)  # absorber carrier density [1/m^3]

# Constants
g0 = 2e-20     # gain coefficient [m^2]
a0 = 1e-20     # absorption coefficient [m^2]
N_tr = 1e24    # transparency carrier density
N_sat = 1.5e24
gamma = 20     # distributed losses [1/m]
dz_m = dz * 1e-6  # convert µm to meters
dt = 0.1e-12   # time step for carrier update [s]

# Carrier lifetimes
tau_g = 1e-9   # [s]
tau_a = 0.5e-9
I_g = 10e-3    # injection current [A]
q = 1.6e-19    # electron charge
V_g = 1e-18    # active volume [m^3]
Gamma_g = 1
Gamma_a = 1

# --- Simulation (one round trip) ---
for i in range(Nz):
    idx = i

    # Get current E, N_g, N_a at position
    E_i = E[idx]
    I_i = np.abs(E_i)**2

    # Calculate gain and absorption
    g_i = g0 * max(N_g[idx] - N_tr, 0) if gain_mask[idx] else 0
    a_i = a0 * (1 - N_a[idx]/N_sat) if abs_mask[idx] else 0

    # Update field: E(z+dz) = E(z) * exp(gain-loss)
    dE = 0.5 * (g_i - a_i - gamma) * dz_m * E[idx]
    E[idx] += dE

    # Update carrier densities (Euler step)
    if gain_mask[idx]:
        dNg = (I_g / (q * V_g) - N_g[idx]/tau_g - Gamma_g * g_i * I_i) * dt
        N_g[idx] += dNg
    if abs_mask[idx]:
        dNa = (-N_a[idx]/tau_a - Gamma_a * a_i * I_i) * dt
        N_a[idx] += dNa

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(Z, np.abs(E)**2, label='|E(z)|²')
plt.plot(Z, N_g, label='N_g')
plt.plot(Z, N_a, label='N_a')
plt.axvspan(0, 200, color='green', alpha=0.1, label='Gain')
plt.axvspan(200, 300, color='red', alpha=0.1, label='Absorber')
plt.xlabel("Position z (µm)")
plt.legend()
plt.title("One Round Trip Along Cavity")
plt.grid()
plt.show()
