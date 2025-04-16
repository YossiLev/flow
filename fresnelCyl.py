import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.interpolate import interp1d

def cylindrical_abcd_propagation(U1_r, r1, r2, wavelength, abcd):
    """
    Compute the field U2(r2) after ABCD propagation of a cylindrically symmetric field U1(r1)
    using a Hankel-like integral.

    Parameters:
        U1_r:      array_like, complex
                   Input field sampled on radial grid r1
        r1:        array_like
                   Radial coordinates of input field [meters]
        r2:        array_like
                   Radial coordinates of output field [meters]
        wavelength: float
                   Wavelength in meters
        abcd:      tuple (A, B, C, D)
                   ABCD matrix parameters of the optical system

    Returns:
        U2_r: array_like, complex
              Output field sampled on radial grid r2
    """
    A, B, C, D = abcd
    k = 2 * np.pi / wavelength

    U2_r = np.zeros_like(r2, dtype=np.complex128)
    dr1 = np.gradient(r1)

    for i, r2_i in enumerate(r2):
        J = j0((k * np.outer(r1, r2[i])) / B)
        phase_in = np.exp(1j * k * A * r1**2 / (2 * B))
        integrand = U1_r * phase_in * r1 * J[:, 0]
        integral = np.sum(integrand * dr1)
        prefactor = np.exp(1j * k * D * r2_i**2 / (2 * B)) / (1j * wavelength * B)
        U2_r[i] = prefactor * integral

    return U2_r

# Define input field: Gaussian beam
def gaussian_beam(r, w0):
    return np.exp(-r**2 / w0**2)

# Parameters
wavelength = 633e-9        # 633 nm
z = 0.1                    # propagation distance (10 cm)
w0 = 1e-3                  # beam waist (1 mm)
A, B, C, D = 1, z, 0, 1    # free space propagation ABCD

# Radial grids
r1 = np.linspace(0, 5e-3, 300)  # input grid (0 to 5 mm)
r2 = r1                         # output grid (same spacing)
U1_r = gaussian_beam(r1, w0)

# Compute propagated field
U2_r = cylindrical_abcd_propagation(U1_r, r1, r2, wavelength, (A, B, C, D))

# Plot
plt.figure(figsize=(8, 4))
plt.plot(r1 * 1e3, np.abs(U1_r), label='|U1(r)|')
plt.plot(r2 * 1e3, np.abs(U2_r), label='|U2(r)| after propagation')
plt.xlabel('r (mm)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.title('Cylindrical ABCD Beam Propagation')
plt.tight_layout()
plt.show()
