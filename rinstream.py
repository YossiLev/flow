import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Constants
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
I = 1.0  # Current in the loop (Amps)
R = 1.0  # Radius of the current loop (meters)

# Magnetic field components in cylindrical coordinates
def magnetic_field_cylindrical(s, pos):
    rho, phi, z = pos

    [rh1, z1] = [-z, rho - 1]
    norm1 = np.sqrt(rh1 * rh1 + z1 * z1)
    den1 = 10 / np.pow(norm1, 3)
    rh1 *= den1
    z1 *= den1

    [rh2, z2] = [-z, rho + 1]
    norm2 = np.sqrt(rh2 * rh2 + z2 * z2)
    den2 = 10 / np.pow(norm2, 3)
    rh2 *= den2
    z2 *= den2
    
    return rh1 - rh2, 0.0, z1 - z2

def magnetic_field_cylindricalo(s, pos):
    rho, phi, z = pos
    r2 = rho**2 + z**2
    denom = (R**2 + r2)**(3/2)
    
    # Avoid singularities
    if denom == 0 or rho == 0:
        return [0, 0, 0]

    # Compute field components
    B_rho = (mu_0 * I * R**2 * z) / (2 * denom)
    B_phi = (mu_0 * I * R**2 * rho) / (2 * denom)  # Important! Phi component must be included
    B_z = (mu_0 * I * R**2 / 2) * (1 / denom + z**2 / denom)
    
    # Compute magnitude for normalization
    B_mag = np.sqrt(B_rho**2 + B_phi**2 + B_z**2)
    
    # Field line evolution: d(rho)/ds, d(phi)/ds, d(z)/ds
    return [B_rho / B_mag, B_phi / B_mag, B_z / B_mag]

# Function to integrate field lines
def integrate_field_line(initial_pos, t_max=5, steps=1000):
    print(initial_pos[0], t_max, steps)
    t_eval = np.linspace(0, t_max, steps)
    sol = solve_ivp(magnetic_field_cylindrical, [0, t_max], initial_pos, t_eval=t_eval, method='RK45')
    return sol.y  # Returns rho, phi, z arrays

# Initial positions (starting near the current loop)
x_vals = np.linspace(1.4, 4., 10) 
initial_positions = [[x, 0, 0] for x in x_vals]

# 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

for pos in initial_positions:
    rho, phi, z = integrate_field_line(pos, 0.5 * np.pow(pos[0] - 1, 2.9))
    
    # Convert cylindrical (rho, phi, z) to Cartesian (x, y, z)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    # Plot field lines
    ax.plot(x, y, z, linewidth=1)

# Draw the current loop (as a red circle in the xy-plane)
theta_loop = np.linspace(0, 2 * np.pi, 100)
x_loop = R * np.cos(theta_loop)
y_loop = R * np.sin(theta_loop)
z_loop = np.zeros_like(theta_loop)
ax.plot(x_loop, y_loop, z_loop, 'r', linewidth=2, label="Current Loop")

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Closed Magnetic Field Lines Around a Current Loop")
plt.legend()
plt.show()
