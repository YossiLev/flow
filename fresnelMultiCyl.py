import numpy as np
from scipy.special import j0
import matplotlib.pyplot as plt


def cylindrical_abcd_propagation_batch(U1_batch, r, wavelength, abcd):
    """
    Vectorized ABCD propagation of multiple cylindrically symmetric field profiles
    using Hankel-like integral.

    Parameters:
        U1_batch: ndarray, shape (N_profiles, N_r)
                  Input fields sampled over radial grid r
        r:        ndarray, shape (N_r,)
                  Radial coordinates (in meters), linearly spaced
        wavelength: float
                  Wavelength in meters
        abcd:     tuple (A, B, C, D)
                  ABCD matrix of the system

    Returns:
        U2_batch: ndarray, shape (N_profiles, N_r)
                  Output fields sampled over the same radial grid r
    """
    A, B, C, D = abcd
    k = 2 * np.pi / wavelength
    N_profiles, N_r = U1_batch.shape

    # Assume linear spacing of r
    dr = r[1] - r[0]
    
    # Precompute kernel matrix: J0(k r1 r2 / B)
    r1 = r.reshape(1, -1)
    r2 = r.reshape(-1, 1)
    kernel = j0(k * r1 * r2 / B)  # shape (N_r, N_r)

    # Precompute phase and prefactor
    phase_input = np.exp(1j * k * A * r / (2 * B) * r)         # shape (N_r,)
    phase_output = np.exp(1j * k * D * r / (2 * B) * r)        # shape (N_r,)
    prefactor = phase_output / (1j * wavelength * B)           # shape (N_r,)

    # Multiply each input profile by phase_input and r
    U1_weighted = U1_batch * (phase_input * r)                 # shape (N_profiles, N_r)

    # Matrix multiplication: (N_profiles, N_r) @ (N_r, N_r)
    U2_batch = U1_weighted @ kernel.T                          # shape (N_profiles, N_r)

    # Multiply each row of result by prefactor
    U2_batch *= dr * prefactor                                 # broadcasting

    return U2_batch

# Parameters
wavelength = 633e-9
z = 8.59
A, B, C, D = 1, z, 0, 1

# Radial grid
N_r = 300
r = np.linspace(0, 5e-3, N_r)

# Example: 3 Gaussian beams with different waists
w0_list = [1e-3, 0.5e-3, 1.5e-3]
U1_batch = np.array([np.exp(-r**2 / w0**2) for w0 in w0_list])

# Propagate all
U2_batch = cylindrical_abcd_propagation_batch(U1_batch, r, wavelength, (A, B, C, D)) * 12.6

# Plotting
plt.figure(figsize=(8, 4))
for i, w0 in enumerate(w0_list):
    plt.plot(r * 1e3, np.abs(U1_batch[i]), label=f"b0 = {w0*1e3:.1f} mm")
    plt.plot(r * 1e3, np.abs(U2_batch[i]), label=f"w0 = {w0*1e3:.1f} mm")
plt.xlabel("r (mm)")
plt.ylabel("Amplitude |U2(r)|")
plt.grid(True)
plt.legend()
plt.title("Cylindrical ABCD Propagation")
plt.tight_layout()
plt.show()