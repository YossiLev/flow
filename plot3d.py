import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate several 2D slices
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# We'll vary the function slightly for each z-level
z_levels = np.linspace(0, 10, 10)  # 10 slices

for z in z_levels:
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) * np.cos(z/2)  # change the slice by z
    ax.plot_surface(X, Y, Z + z, alpha=0.6, cmap='viridis')  # offset in z

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D plot made of multiple 2D slices")
plt.show()
