import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def cartesian(arrays, dtype=None, out=None):
    arrays = [np.asarray(x) for x in arrays]
    if dtype is None:
        dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def xyz(at):
    [r, theta, phi] = at
    return (np.cos(phi) * np.cos(theta) * r,
           np.sin(phi) * np.cos(theta) * r,
           np.sin(theta) * r)
# Define electric field function for a point charge
def electric_field(x, y, z, q=1, k=1):
    r_squared = x**2 + y**2 + z**2
    r = np.sqrt(r_squared) + 1e-10  # Avoid division by zero
    E_x = k * q * x / r_squared
    E_y = k * q * y / r_squared
    E_z = k * q * z / r_squared
    return np.array([E_x, E_y, E_z])

# Define magnetic field function for a dipole moment along the z-axis
def magnetic_field(x, y, z, m=1, k=1):
    r_squared = x**2 + y**2 + z**2
    r = np.sqrt(r_squared) + 1e-10  # Avoid division by zero
    B_x = (3 * x * z) / r**(3/2)
    B_y = (3 * y * z) / r**(3/2)
    B_z = (2 * z**2 - x**2 - y**2) / r**(3/2)
    return np.array([B_x, B_y, B_z])

# Define Poynting vector S = E x B
def poynting_vector(x, y, z):
    E = electric_field(x, y, z)
    B = magnetic_field(x, y, z)
    S = np.cross(E, B, axis=0)
    return S

# Integrate field lines
def compute_streamlines(field_func, seed_points, steps=100, step_size=0.1):
    streamlines = []
    for seed in seed_points:
        path = [seed]
        point = np.array(seed, dtype=float)
        for _ in range(steps):
            field = field_func(*point)
            field /= np.linalg.norm(field) + 1e-10  # Normalize
            point += field * step_size
            path.append(point.copy())
        streamlines.append(np.array(path))
    return streamlines

# Define seed points for streamlines
num_seeds = 10
phi = np.linspace(0, 2 * np.pi, num_seeds)
theta = np.linspace(- np.pi, np.pi, 6)[1:]
x_seeds = np.cos(phi)
y_seeds = np.sin(phi)
z_seeds = np.zeros_like(phi)
listx = [xyz(at) for at in cartesian([[1.0], theta, phi], 'object')]

electric_seed_points = listx

# num_seeds = 10
# theta = np.linspace(0, 2 * np.pi, num_seeds)
# x_seeds = np.cos(theta)
# y_seeds = np.sin(theta)
# z_seeds = np.zeros_like(theta)
listx = [xyz(at) for at in cartesian([[0.4], [0.05], [0.0]], 'object')]
magnetic_seed_points = listx

num_seeds = 10
theta = np.linspace(0, 2 * np.pi, num_seeds)
x_seeds = np.cos(theta)
y_seeds = np.sin(theta)
z_seeds = np.zeros_like(theta)
poynting_seed_points = list(zip(x_seeds, y_seeds, z_seeds))
listx = [xyz(at) for at in cartesian([[1.0, 3.0, 5.0], theta, [0.0]], 'object')]
poynting_seed_points = listx

# Compute electric, magnetic, and Poynting vector streamlines
electric_streamlines = compute_streamlines(electric_field, electric_seed_points)
magnetic_streamlines = compute_streamlines(magnetic_field, magnetic_seed_points)
poynting_streamlines = compute_streamlines(poynting_vector, poynting_seed_points)

# Create figure
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Plot electric field streamlines
# for streamline in electric_streamlines:
#     ax.plot3D(streamline[:, 0], streamline[:, 1], streamline[:, 2], color='blue', linewidth=1)

# Plot magnetic field streamlines
for streamline in magnetic_streamlines:
    ax.plot3D(streamline[:, 0], streamline[:, 1], streamline[:, 2], color='green', linewidth=1)

# Plot Poynting vector streamlines
# for streamline in poynting_streamlines:
#     ax.plot3D(streamline[:, 0], streamline[:, 1], streamline[:, 2], color='purple', linewidth=1)

# Draw charge location
ax.scatter(0, 0, 0, color='red', s=100, label='Point Charge')

# Labels and styling
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Electric, Magnetic, and Poynting Vector Streamlines of a Point Charge with Magnetic Moment")
ax.legend()
plt.show()
