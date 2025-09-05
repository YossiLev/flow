# Plot S/u along the x-axis for a rotating charged ring with:
# R = reduced Compton wavelength of electron = ħ / (m_e c)
# ω chosen so that ω R = c  =>  ω = c / R
# Q = -e (electron charge)
#
# We'll show:
# 1) E_x(x) and H_z(x)
# 2) Ratio |E×H| / (0.5*eps0*E^2 + 0.5*mu0*H^2)
#
# Range: x from 0 to 10 R (avoid the singular point exactly at x=R)

import numpy as np
import matplotlib.pyplot as plt

# Physical constants (CODATA-like values)
eps0 = 8.8541878128e-12
mu0 = 4*np.pi*1e-7
c = 299792458.0
hbar = 1.054571817e-34
m_e = 9.1093837015e-31
e = 1.602176634e-19

# Parameters per user request
R = hbar / (m_e * c)           # ~3.8615926764e-13 m
omega = c / R                   # so that omega*R = c
Q = -e                          # electron charge (sign affects E/H signs, not |S|)

# Derived
I = Q * omega / (2*np.pi)

# Integration discretization
Nphi = 4000
phi = np.linspace(0, 2*np.pi, Nphi, endpoint=False)
dphi = 2*np.pi / Nphi

# Source ring geometry
x_src = R * np.cos(phi)
y_src = R * np.sin(phi)

# dl vector components
dlx = -R * np.sin(phi) * dphi
dly =  R * np.cos(phi) * dphi
dq = Q / (2*np.pi) * dphi

# Evaluation points along x (avoid exactly x=R to prevent divergence)
x_vals = np.linspace(0.0, 10.0*R, 600)
# Nudge any point too close to R
x_vals = np.where(np.isclose(x_vals, R, rtol=0, atol=1e-9*R), x_vals + 1e-9*R, x_vals)

E_x = np.zeros_like(x_vals)
B_z = np.zeros_like(x_vals)

for i, x in enumerate(x_vals):
    rx = x - x_src
    ry = - y_src
    r2 = rx*rx + ry*ry
    r = np.sqrt(r2)
    r3 = r2 * r
    # Electric field (Coulomb integral of static ring)
    Ex = (1/(4*np.pi*eps0)) * np.sum(dq * rx / r3)
    # Magnetic field from Biot–Savart (steady current I)
    cz = dlx*ry - dly*rx
    Bz = (mu0*I/(4*np.pi)) * np.sum(cz / r3)
    E_x[i] = Ex
    B_z[i] = Bz

H_z = B_z / mu0

# Compute S/u ratio
S_mag = np.abs(E_x * H_z)
u = 0.5 * eps0 * E_x**2 + 0.5 * mu0 * H_z**2
ratio = np.full_like(x_vals, np.nan)
mask = u > 0
ratio[mask] = S_mag[mask] / u[mask]

# Report the actual parameter numbers for transparency
R_val = R
omega_val = omega
Q_val = Q
I_val = I

print("Parameters used:")
print(f"R = ħ/(m_e c) = {R_val:.6e} m")
print(f"ω = c/R = {omega_val:.6e} rad/s  (so ωR = c = {c:.6e} m/s)")
print(f"Q = -e = {Q_val:.6e} C")
print(f"Effective current I = Q ω / (2π) = {I_val:.6e} A")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of subplots

# Plot E_x
axes[0, 0].plot(x_vals/R, E_x)
axes[0, 0].set_xlabel('x / R')
axes[0, 0].set_ylabel('E_x (V/m)')
axes[0, 0].set_title('E_x on x-axis (electron ring, R = ħ/m_ec, ωR = c)')

# Plot H_z
axes[0, 1].plot(x_vals/R, H_z)
axes[0, 1].set_xlabel('x / R')
axes[0, 1].set_ylabel('H_z (A/m)')
axes[0, 1].set_title('H_z on x-axis (electron ring, R = ħ/m_ec, ωR = c)')

# Plot ratio S/u (linear)
axes[1, 0].plot(x_vals/R, ratio)
axes[1, 0].set_xlabel('x / R')
axes[1, 0].set_ylabel('S / u (m/s)')
axes[1, 0].set_title('Poynting magnitude / energy density along x-axis')

# clip y-axis to 98th percentile to keep curve visible
# finite_vals = ratio[np.isfinite(ratio)]
# if finite_vals.size > 0:
#     ymax = np.nanpercentile(finite_vals, 98)
#     plt.ylim(0, ymax if ymax>0 else None)
# plt.tight_layout()
# plt.show()

# Plot log10 of ratio
logratio = np.full_like(ratio, np.nan)
good = np.isfinite(ratio) & (ratio>0)
logratio[good] = np.log10(ratio[good])
axes[1, 1].plot(x_vals/R, logratio)
axes[1, 1].set_xlabel('x / R')
axes[1, 1].set_ylabel('log10(S / u)')
axes[1, 1].set_title('log10 of Poynting magnitude / energy density')
for ax in axes.flat:
    ax.grid(True)

plt.tight_layout()
plt.show()
