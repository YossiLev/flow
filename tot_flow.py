# Correct the gradient ordering and replot the vector field as the negative gradient of phi.
import numpy as np
import matplotlib.pyplot as plt

# Geometry parameters (same as before)
R_outer = 12.0
R_torus_major = 3.0
R_torus_minor = 0.4

# Grid for xz-plane
nx, nz = 300, 300
x_vals = np.linspace(-R_outer, R_outer, nx)
z_vals = np.linspace(-R_outer, R_outer, nz)
X, Z = np.meshgrid(x_vals, z_vals)
Y = np.zeros_like(X)

# Domain masks
r_outer = np.sqrt(X**2 + Y**2 + Z**2)
rho_xy = np.sqrt(X**2 + Y**2)
torus_eq = (rho_xy - R_torus_major)**2 + Z**2
inside_sphere = r_outer <= R_outer
outside_torus = torus_eq >= R_torus_minor**2
domain_mask = inside_sphere & outside_torus

# Potential (same smooth interpolation)
dist_sphere = R_outer - r_outer
dist_torus = np.sqrt(torus_eq) - R_torus_minor
phi = - dist_sphere / (dist_sphere + dist_torus + 1e-9)
phi[~domain_mask] = np.nan  # mask outside domain for clarity

# Correct gradient calculation: np.gradient returns gradient along each axis in array order.
# phi has shape (nz, nx) corresponding to axes (z, x). So gradients are (dphi/dz, dphi/dx).
dphidz, dphidx = np.gradient(phi, z_vals, x_vals, edge_order=2)
Vx = -dphidx
Vz = -dphidz

# Mask velocities outside domain (set to zero to avoid NaNs in streamplot)
Vx_masked = np.where(domain_mask, Vx, 0.0)
Vz_masked = np.where(domain_mask, Vz, 0.0)

# Normalize vector field for cleaner arrows in quiver
speed = np.sqrt(Vx_masked**2 + Vz_masked**2)
nonzero = speed > 1e-12
Vx_norm = np.zeros_like(Vx_masked)
Vz_norm = np.zeros_like(Vz_masked)
Vx_norm[nonzero] = Vx_masked[nonzero] / speed[nonzero]
Vz_norm[nonzero] = Vz_masked[nonzero] / speed[nonzero]

# Plot: potential background + true gradient streamlines (negative gradient points from sphere->torus)
fig, ax = plt.subplots(figsize=(8,8))
cf = ax.contourf(X, Z, phi, levels=50, cmap='viridis')
plt.colorbar(cf, ax=ax, label='Potential φ')

# Streamlines using the true velocity field (negative gradient)
# Use density adjusted to domain size; mask outside by setting vectors zero.
strm = ax.streamplot(x_vals, z_vals, Vx_masked, Vz_masked, 
                     color='white', linewidth=0.5, density=1.4, arrowsize=0.0,
                     broken_streamlines=False)

# Draw boundaries
outer = plt.Circle((0,0), R_outer, edgecolor='red', fill=False, lw=2, label='Outer sphere (slice)')
torus_r = plt.Circle(( R_torus_major, 0), R_torus_minor, edgecolor='orange', fill=False, lw=2, label='Torus (slice)')
torus_l = plt.Circle((-R_torus_major, 0), R_torus_minor, edgecolor='orange', fill=False, lw=2)
ax.add_artist(outer); ax.add_artist(torus_r); ax.add_artist(torus_l)

ax.set_aspect('equal')
ax.set_xlim(-R_outer, R_outer)
ax.set_ylim(-R_outer, R_outer)
ax.set_xlabel('x'); ax.set_ylabel('z')
ax.set_title('Corrected velocity field: v = -∇φ (xz-plane slice)')
ax.legend(loc='upper right')

plt.show()
