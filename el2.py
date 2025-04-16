import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

# Constants
mu_0 = 4 * np.pi * 1e-7  # permeability of free space (H/m)
I = 1.0  # Current in the ring (Amps)
R = 1.0  # Radius of the current ring (meters)

# Biot-Savart Law to calculate the magnetic field at a point due to a small current element
def magnetic_field(r, t, I=I, R=R):
    # Parametric angle of the current element
    theta = t
    # Position of the current element in the ring
    x1 = R * np.cos(theta)
    y1 = R * np.sin(theta)
    z1 = 0

    # Position vector from the current element to the observation point
    r_vec = r - np.array([x1, y1, z1])
    r_mag = np.linalg.norm(r_vec)

    # Magnetic field due to current element (Biot-Savart law)
    B = (mu_0 * I / (4 * np.pi)) * (np.cross([0, 0, 1], r_vec)) / (r_mag**3)
    
    return B

# Function to calculate the streamlines (ODE system for streamline propagation)
def model(r, t):
    # Magnetic field at position r
    B = magnetic_field(r, t)
    # Streamline direction is perpendicular to the magnetic field
    return B

# Create a grid of points in 3D space
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
z = np.linspace(-2, 2, 10)
X, Y, Z = np.meshgrid(x, y, z)

# Flatten the grid
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# Solve ODE for each initial position to trace the streamlines
streamlines = []
for pos in positions:
    # Solve the ODE starting from each point
    t = np.linspace(0, 20, 500)
    trajectory = odeint(model, pos, t)
    streamlines.append(trajectory)

# Plot the streamlines in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each streamline
for streamline in streamlines:
    ax.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2], lw=0.7)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Magnetic Streamlines Around a Current Ring')

plt.show()
