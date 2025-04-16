import numpy as np
from scipy.special import ellipk, ellipe
import matplotlib.pyplot as plt

# Constants
mu_0 = 1.0  # Permeability of free space (simplified units)
I = 1.0     # Current in the ring (counterclockwise)
a = 1.0     # Radius of the ring

# Magnetic field components for a ring current
def magnetic_field(rho, z):
    """Calculate B_rho and B_z for a ring current at (rho, z)."""
    # Avoid singularities at rho = 0 and at the ring (rho = a, z = 0)
    rho = np.where(rho == 0, 1e-10, rho)  # Small offset to avoid division by zero
    mask = (np.abs(rho - a) < 1e-5) & (np.abs(z) < 1e-5)
    rho = np.where(mask, rho + 1e-5, rho)  # Avoid the ring itself
    
    k2 = 4 * rho * a / ((rho + a)**2 + z**2)
    k = np.sqrt(k2)
    K = ellipk(k2)  # Complete elliptic integral of the first kind
    E = ellipe(k2)  # Complete elliptic integral of the second kind
    
    denom = np.sqrt((rho + a)**2 + z**2)
    factor = mu_0 * I / (2 * np.pi)
    
    # Common terms
    term1 = -K
    term2 = (rho**2 + a**2 + z**2) / ((rho - a)**2 + z**2) * E
    term3 = (rho**2 - a**2 + z**2) / ((rho - a)**2 + z**2) * E
    
    # B_rho component
    B_rho = factor * z / (rho * denom) * (term1 + term2)
    
    # B_z component
    B_z = factor / denom * (K + term3)
    
    return B_rho, B_z

def magnetic_fielde(rho, z):
    """Calculate B_rho and B_z for a ring current at (rho, z)."""
    # Avoid singularities at rho = 0 and at the ring (rho = a, z = 0)
    [rh1, z1] = [-z, rho - 1]
    norm1 = np.sqrt(rh1 * rh1 + z1 * z1)
    den1 = 1.0 / pow(norm1, 3)
    rh1 *= den1
    z1 *= den1

    [rh2, z2] = [-z, rho + 1]
    norm2 = np.sqrt(rh2 * rh2 + z2 * z2)
    den2 = 1.0 / pow(norm2, 3)
    rh2 *= den2
    z2 *= den2
    
    return rh1 - rh2, z1 - z2

# Create a grid in the rho-z plane
rho = np.linspace(-2, 2, 20)  # Radial distance from 0 to 2
z = np.linspace(-1.5, 1.5, 20)  # Axial distance from -1.5 to 1.5
RHO, Z = np.meshgrid(rho, z)

# Compute magnetic field components on the grid
B_rho, B_z = magnetic_fielde(RHO, Z)

# Normalize the vectors for better visualization (optional)
B_magnitude = np.sqrt(B_rho**2 + B_z**2)
B_rho_norm = B_rho / B_magnitude * np.pow(B_magnitude, 0.3) / 2
B_z_norm = B_z / B_magnitude * np.pow(B_magnitude, 0.3) / 2

# Plotting
plt.figure(figsize=(10, 8))
plt.quiver(RHO, Z, B_rho_norm, B_z_norm, scale=30, color='b', label='Normalized B-field')
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)  # z = 0 plane
plt.axvline(a, color='r', linestyle='--', label='Ring current (ρ = 1)')  # Ring radius
plt.xlabel('ρ (radial distance)')
plt.ylabel('z (axial distance)')
plt.title('Magnetic Field Vectors in the ρz-Plane for a Ring Current')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Equal aspect ratio for proper visualization
plt.show()

# Verify field direction at z = 0 for rho < 1 and rho > 1
rho_test = np.array([0.5, 1.5])
z_test = np.array([0.0, 0.0])
B_rho_test, B_z_test = magnetic_field(rho_test, z_test)
print(f"At ρ = 0.5, z = 0: B_rho = {B_rho_test[0]:.4f}, B_z = {B_z_test[0]:.4f}")
print(f"At ρ = 1.5, z = 0: B_rho = {B_rho_test[1]:.4f}, B_z = {B_z_test[1]:.4f}")