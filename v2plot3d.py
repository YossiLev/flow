import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample x values for all the 1D lines
x = np.linspace(0, 10, 100)

# Create several 1D functions
def make_line(offset):
    return np.sin(x + offset)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot several 1D lines side by side along the Y axis (or Z if you prefer)
for i, offset in enumerate(np.linspace(0, 5, 20)):
    y = make_line(offset)
    z = np.full_like(x, offset)  # constant z for each line
    ax.plot(x, y, z, color='blue')  # lines along x-y, stacked along z

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D plot made from multiple 1D lines")
plt.show()
