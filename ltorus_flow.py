import numpy as np
import matplotlib.pyplot as plt

# Torus parameters
R0 = 1.0   # Major radius
r0 = 0.4   # Minor radius
V0 = 1.0   # Toroidal velocity magnitude

# Grid in X-Z plane
x = np.linspace(-2, 2, 300)
z = np.linspace(-1.5, 1.5, 300)
X, Z = np.meshgrid(x, z)

# Distance from torus sleeves at x = ±R0
dist1 = np.sqrt((X - R0)**2 + Z**2)
dist2 = np.sqrt((X + R0)**2 + Z**2)

# Velocity profile (only y-component)
V1 = V0 * np.exp(-(dist1 / r0)**2)
V2 = -V0 * np.exp(-(dist2 / r0)**2)
Vy = V1 + V2

# Vorticity components (in-plane, X and Z)
theta1 = np.arctan2(Z, X - R0)
theta2 = np.arctan2(Z, X + R0)

# Vorticity vectors
W1_x = -np.sin(theta1) * V1
W1_z = np.cos(theta1) * V1
W2_x = -np.sin(theta2) * V2
W2_z = np.cos(theta2) * V2
Wx = W1_x + W2_x
Wz = W1_z + W2_z

# Lamb vector: L = V x W
# Since V = (0, Vy, 0), W = (Wx, 0, Wz), then:
# L = Vy * (-Wz, 0, Wx)
Lx = -Vy * Wz
Lz = Vy * Wx

# Plotting
fig, ax = plt.subplots(figsize=(9, 6))

# Background: magnitude of Lamb vector
L_mag = np.sqrt(Lx**2 + Lz**2)
contour = ax.contourf(X, Z, L_mag, levels=50, cmap='inferno')
cbar = plt.colorbar(contour, ax=ax, label='|Lamb Vector|')

# Arrows: Lamb vector field in X-Z plane
skip = (slice(None, None, 6), slice(None, None, 6))
ax.quiver(X[skip], Z[skip], Lx[skip], Lz[skip], color='lime', pivot='middle', scale=10, label='Lamb Vector Field')

# Draw torus sleeves
theta = np.linspace(0, 2*np.pi, 200)
for xc in [-R0, R0]:
    ax.plot(xc + r0 * np.cos(theta), r0 * np.sin(theta), 'k--', linewidth=1)

ax.set_title('XZ Slice of Torus: Lamb Vector Field (V × ω)')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_aspect('equal')
plt.grid(True)
plt.legend()
plt.show()
