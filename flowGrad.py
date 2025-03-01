import numpy as np
import matplotlib.pyplot as plt

# Define the potential function: phi = Vx - (Q / (2π)) ln(r)
def potential(r, theta, V=0.5, Q=3):
    phi = V * r * np.cos(theta) - (Q / (2 * np.pi)) * np.log(r + 1e-10)  # Avoid log(0)
    return phi

# Compute the velocity field (gradient of the potential)
def velocity_field(r, theta, V=0.5, Q=3):
    v_r = V * np.cos(theta) - (Q / (2 * np.pi * r))
    v_theta = -V * np.sin(theta)
    return v_r, v_theta

# Grid definition in polar coordinates
theta = np.linspace(0, 2*np.pi, 28)
r = np.linspace(0.4, 2, 10)  # Avoid r=0 singularity
R, Theta = np.meshgrid(r, theta)

# Convert polar to Cartesian for plotting
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# Compute potential values
Phi = potential(R, Theta)

# Compute velocity field
U, V = velocity_field(R, Theta)

# Convert velocity field to Cartesian components
U_cart = U * np.cos(Theta) - V * np.sin(Theta)
V_cart = U * np.sin(Theta) + V * np.cos(Theta)

# Plot potential lines
plt.figure(figsize=(6,6))
# Add a full circle in the center with radius 0.5
circle = plt.Circle((0, 0), 0.4, color='blue', alpha=0.2, fill=True, linewidth=2)
plt.gca().add_patch(circle)
contour = plt.contour(X, Y, Phi, levels=20, cmap="plasma")
plt.colorbar(contour, label="Potential φ")
plt.scatter(0, 0, color="red", marker="x", label="Sink location")

# Plot velocity field as arrows
plt.quiver(X, Y, U_cart, V_cart, color="black", alpha=0.6, scale=30)

# Labels and styling
plt.xlabel("x")
plt.ylabel("y")
plt.title("Equipotential Lines and Velocity Field of a Sink + Uniform Flow")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)

# Show plot
plt.show()
