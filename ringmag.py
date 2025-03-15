import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp  # For elliptic integral

# Constants
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
I = 1.0  # Current in the loop (Amps)
R = 1.0  # Radius of the loop

# Define function for A_phi
def A_phi(rho, z):
    if rho == 0:  # Avoid division by zero
        return 0
    k = 2 * np.sqrt(R * rho) / np.sqrt((R + rho) ** 2 + z ** 2)
    return (mu_0 * I / np.pi) * np.sqrt(R / rho) * sp.ellipk(k**2)

# Generate a grid of (rho, z) values
rho_vals = np.linspace(0.1, 2.5, 100)  # Avoid rho = 0 for singularity
z_vals = np.linspace(-2, 2, 100)
RHO, Z = np.meshgrid(rho_vals, z_vals)

# Compute A_phi on the grid
A_PHI = np.vectorize(A_phi)(RHO, Z)

# Plot the magnetic vector potential
plt.figure(figsize=(8, 6))
contour = plt.contourf(RHO, Z, A_PHI, levels=30, cmap="plasma")

plt.grid()
plt.show()
