import numpy as np
import matplotlib.pyplot as plt

# Torus parameters
R0 = 1.0   # Major radius (center of torus tube)
r0 = 0.4   # Minor radius (radius of tube cross-section)
V0 = 1.0   # Velocity magnitude in torus

# Grid in X-Z plane
x = np.linspace(-2, 2, 300)
z = np.linspace(-1.5, 1.5, 300)
X, Z = np.meshgrid(x, z)

# Distance from two torus tube centers at (±R0, 0)
dist1 = np.sqrt((X - R0)**2 + Z**2)
dist2 = np.sqrt((X + R0)**2 + Z**2)

# Assume velocity exists only inside torus tubes, decays outward
V1 = V0 * np.exp(-(dist1 / r0)**2)
V2 = -V0 * np.exp(-(dist2 / r0)**2)  # Opposite direction for other side
velocity_magnitude = V1 + V2

# Vorticity field: circulating around sleeves (in-plane)
# Compute tangents to circles at (±R0, 0)
theta1 = np.arctan2(Z, X - R0)
theta2 = np.arctan2(Z, X + R0)

# Local circulation directions (qualitative)
W1_x = -np.sin(theta1) * V1
W1_z = np.cos(theta1) * V1
W2_x = -np.sin(theta2) * V2
W2_z = np.cos(theta2) * V2

# Total in-plane vorticity
W_x = W1_x + W2_x
W_z = W1_z + W2_z

# Plot
fig, ax = plt.subplots(figsize=(9, 6))

# Show velocity magnitude as background (into/out of plane)
contour = ax.contourf(X, Z, velocity_magnitude, levels=50, cmap='plasma')
cbar = plt.colorbar(contour, ax=ax, label='Velocity (toroidal direction)')

# Show vorticity field as arrows
skip = (slice(None, None, 6), slice(None, None, 6))
ax.quiver(X[skip], Z[skip], W_x[skip], W_z[skip], color='cyan', pivot='middle', scale=20, label='Vorticity Field')

# Draw two torus cross-sections (sleeves)
theta = np.linspace(0, 2*np.pi, 200)
for xc in [-R0, R0]:
    ax.plot(xc + r0 * np.cos(theta), r0 * np.sin(theta), 'k--', linewidth=1)

ax.set_title('XZ Slice of Torus: Toroidal Velocity and Poloidal Vorticity')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_aspect('equal')
plt.grid(True)
plt.legend()
plt.show()
