# Compute and plot the Poynting magnitude divided by energy density along x-axis for the rotating charged ring.
# Using same parameters as before: R=1 m, Q=1e-6 C, omega=1000 rad/s.
# S_mag = |E x H| = |E_x * H_z| (here E along x, H along z)
# u = 0.5 * eps0 * E^2 + 0.5 * mu0 * H^2
# We plot ratio = S_mag / u (units: m/s)
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
eps0 = 8.8541878128e-12
mu0 = 4*np.pi*1e-7

# Parameters
R = 1.0
Q = 1e-6
omega = 1000.0
I = Q * omega / (2*np.pi)

# Integration around ring
Nphi = 2000
phi = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
dphi = 2*np.pi / Nphi
x_src = R * np.cos(phi)
y_src = R * np.sin(phi)
dlx = -R * np.sin(phi) * dphi
dly =  R * np.cos(phi) * dphi
dq = Q / (2*np.pi) * dphi

# Evaluation points
x_vals = np.linspace(0.0, 10.0*R, 400)
E_x = np.zeros_like(x_vals)
B_z = np.zeros_like(x_vals)

for i, x in enumerate(x_vals):
    rx = x - x_src
    ry = - y_src
    r2 = rx*rx + ry*ry
    r = np.sqrt(r2)
    r3 = r2 * r
    Ex = (1/(4*np.pi*eps0)) * np.sum(dq * rx / r3)
    cz = dlx*ry - dly*rx
    Bz = (mu0*I/(4*np.pi)) * np.sum(cz / r3)
    E_x[i] = Ex
    B_z[i] = Bz

H_z = B_z / mu0

# Compute Poynting magnitude and energy density and their ratio
S_mag = np.abs(E_x * H_z)  # |E x H| (magnitude)
u = 0.5 * eps0 * E_x**2 + 0.5 * mu0 * H_z**2  # energy density (J/m^3)
ratio = np.full_like(x_vals, np.nan)
mask = u > 0
ratio[mask] = S_mag[mask] / u[mask]

# Plot ratio (linear) and log10 where defined
plt.figure(figsize=(8,4.5))
plt.plot(x_vals/R, ratio)
plt.xlabel('x / R')
plt.ylabel('S / u (m/s)')
plt.title('Poynting magnitude divided by energy density: |E×H| / u along x-axis')
plt.grid(True)
plt.ylim(0, np.nanpercentile(ratio, 98))  # zoom to avoid huge spikes
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4.5))
logratio = np.full_like(ratio, np.nan)
defined = ~np.isnan(ratio) & (ratio>0)
logratio[defined] = np.log10(ratio[defined])
plt.plot(x_vals/R, logratio)
plt.xlabel('x / R')
plt.ylabel('log10(S / u)')
plt.title('log10 of |E×H| / u along x-axis')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print a few sample numeric values near center, at R, and far away
sample_indices = [0, np.argmin(np.abs(x_vals-R)), np.argmin(np.abs(x_vals-5*R)), -1]
samples = [(x_vals[i], E_x[i], H_z[i], ratio[i]) for i in sample_indices]
samples
