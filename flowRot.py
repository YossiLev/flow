import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Convert to polar coordinates
r = np.sqrt(X**2 + Y**2) + 1e-5  # Avoid division by zero
theta = np.arctan2(Y, X)

# Define velocity components
strength_sink = -1  # Sink strength (negative for inflow)
strength_rotation = 1  # Rotation strength

Vr = strength_sink / r * (r - 0.5) # Radial inflow velocity
Vtheta = strength_rotation / r  # Rotational velocity

# Convert back to Cartesian coordinates
U = Vr * np.cos(theta) - Vtheta * np.sin(theta)
V = Vr * np.sin(theta) + Vtheta * np.cos(theta)

# Plot the streamlines
plt.figure(figsize=(6,6))
plt.streamplot(X, Y, U, V, color='b', density=1.5)
plt.xlabel("X")
plt.ylabel("Y")
#plt.xlim(-2, 2)
#plt.ylim(-2, 2)
#plt.axhline(0, color='k', linewidth=0.5)
#plt.axvline(0, color='k', linewidth=0.5)
#plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
