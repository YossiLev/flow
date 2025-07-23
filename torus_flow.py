import numpy as np
import matplotlib.pyplot as plt

# Torus parameters
R0 = 1.0  # major radius (distance from center to tube center)
r0 = 0.4  # minor radius (tube radius)

# Grid for 2D cross-section (x-y plane, center of torus at origin)
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)

# Convert to polar coordinates w.r.t. center of torus tube
# Distance from the torus ring center (circular cross-section)
distance_from_ring = np.sqrt((np.sqrt(X**2 + Y**2) - R0)**2 + 0**2)

# Velocity profile: flow along the toroidal direction (out of the plane)
# We'll show velocity magnitude as scalar field
V0 = 1.0  # max velocity inside the torus
velocity_magnitude = V0 * np.exp(-(distance_from_ring / r0)**2)

# For plotting: use direction vectors for vorticity (circulating poloidally)
# Compute angle relative to torus tube center (local poloidal direction)
theta = np.arctan2(Y, X)
W_x = -np.sin(theta) * velocity_magnitude
W_y = np.cos(theta) * velocity_magnitude

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Velocity magnitude as background scalar field (flow into/out of the screen)
contour = ax.contourf(X, Y, velocity_magnitude, levels=50, cmap='plasma')
cbar = plt.colorbar(contour, ax=ax, label='Velocity Magnitude (V)')

# Vorticity field as vectors (in the poloidal direction)
skip = (slice(None, None, 5), slice(None, None, 5))  # reduce number of arrows
ax.quiver(X[skip], Y[skip], W_x[skip], W_y[skip], color='cyan', pivot='middle', scale=20, label='Vorticity Field (W)')

# Torus tube outline
circle = plt.Circle((R0, 0), r0, color='k', fill=False, linestyle='--', linewidth=1)
ax.add_artist(circle)

ax.set_title('Cross-section of Torus Flow: Velocity (V) and Vorticity (W)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend()
plt.grid(True)
plt.show()
