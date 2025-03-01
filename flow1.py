import numpy as np
import matplotlib.pyplot as plt

# Define the potential function: phi = Vx - (Q / (2π)) ln(r)
def potential(x, y, V=1, Q=1):
    r = np.sqrt(x**2 + y**2)
    phi = V * x - (Q / (2 * np.pi)) * np.log(r + 1e-10)  # Avoid log(0)
    return phi

# Grid definition
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Compute potential values
Phi = potential(X, Y, V=1, Q=4)

# Plot potential lines
plt.figure(figsize=(6,6))
contour = plt.contour(X, Y, Phi, levels=20, cmap="plasma")
plt.colorbar(contour, label="Potential φ")
plt.scatter(0, 0, color="red", marker="x", label="Sink location")

# Labels and styling
plt.xlabel("x")
plt.ylabel("y")
plt.title("Equipotential Lines of a Sink + Uniform Flow")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)

# Show plot
plt.show()
