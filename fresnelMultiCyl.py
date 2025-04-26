import numpy as np
from scipy.special import j0
import matplotlib.pyplot as plt


def front_power(bin_field, x):
    bin_intencity = np.square(np.abs(bin_field))
    bin_intencity = np.multiply(bin_intencity, x)
    return np.sum(bin_intencity, axis=1)

def compute_total_power(U, r):
    """
    Computes the total optical power of a scalar field U(r)
    under cylindrical symmetry.

    Parameters:
        U: ndarray, shape (N,) or (M, N) for batch
        r: ndarray, shape (N,)

    Returns:
        power: float or ndarray, total power
    """
    dr = r[1] - r[0]
    intensity = np.abs(U)**2
    if U.ndim == 1:
        power = 2 * np.pi * np.sum(intensity * r) * dr
    else:
        power = 2 * np.pi * np.sum(intensity * r[None, :], axis=1) * dr
    return power


def cylindrical_abcd_propagation_K(U1_batch, r, wavelength, abcd):
    A, B, C, D = abcd
    k = 2 * np.pi / wavelength

    # Assume linear spacing of r
    dr = r[1] - r[0]
    
    # Precompute kernel matrix: J0(k r1 r2 / B)
    r1 = r.reshape(1, -1)
    r2 = r.reshape(-1, 1)
    kernel = j0(k * r1 * r2 / B)  # shape (N_r, N_r)

    # Precompute phase and prefactor
    phase_input = np.exp(1j * k * A * r / (2 * B) * r)         # shape (N_r,)
    phase_output = np.exp(1j * k * D * r / (2 * B) * r)        # shape (N_r,)
    prefactor = 2 * np.pi * phase_output / (1j * wavelength * B)           # shape (N_r,)

    # Multiply each input profile by phase_input and r
    U1_weighted = U1_batch * (phase_input * r)                 # shape (N_profiles, N_r)

    # Matrix multiplication: (N_profiles, N_r) @ (N_r, N_r)
    U2_batch = U1_weighted @ kernel.T                          # shape (N_profiles, N_r)

    # Multiply each row of result by prefactor
    U2_batch *= dr * prefactor                                 # broadcasting

    return U2_batch
    # # Phase terms
    # phase_in = np.exp(1j * k * A * r**2 / (2 * B))         # shape (N,)
    # phase_out = np.exp(1j * k * D * r**2 / (2 * B))        # shape (N,)
    # prefactor =  2 * np.pi / (1j * wavelength * B)

    # # Kernel matrix
    # r1 = r.reshape((1, -1))  # input (columns)
    # r2 = r.reshape((-1, 1))  # output (rows)
    # kernel = j0(k * r1 * r2 / B)                            # shape (N, N)

    # # Combine all
    # K = prefactor * (phase_out[:, None] * kernel * phase_in[None, :] * r1 * dr)

    # #U2_batch = U1_batch @ K.T                          # shape (N_profiles, N_r)

    # return K

# Parameters
wavelength = 780e-9
z = 1
A, B, C, D = 1, z, 0, 1


N_r = 512
max_val = 0.000125
r = np.linspace(0, max_val, N_r)
sh = r[1] * 0.5
r = r + sh


abcds = [[-0.8262222222222222, -0.0021001555555555593, 151.11111111111111, -0.8262222222222211]]
abcds = [[0.955999760000005, -0.00029340156436342274, 293.33333333333337, 0.9559997600000021]]
abcds = [[-0.8262222222222222, -0.0021001555555555593, 151.11111111111111, -0.8262222222222211],
         [0.955999760000005, -0.00029340156436342274, 293.33333333333337, 0.9559997600000021],
        #     [1, 0.03, 0, 1],
        #     [1, 0.04, 0, 1],
        #     [1, 0.05, 0, 1],
        #     [1, 0.06, 0, 1],
        #     [1, 0.07, 0, 1],
        #     [1, 0.08, 0, 1],
        #     [1, 0.09, 0, 1],
        #     [1, 0.10, 0, 1]
            ]
        
# K_batch = []
# for abcd in abcds:
#     A, B, C, D = abcd
#     K = cylindrical_abcd_propagation_K(r, wavelength, (A, B, C, D))
#     K_batch.append(K)

# Example: 3 Gaussian beams with different waists
w0_list = np.array([0.000060])
print("Waists (m):", w0_list)
rayligh = np.pi * np.square(w0_list) / wavelength
#print("Rayleigh lengths (m):", rayligh)
target_waists = w0_list * np.sqrt(1 + np.square((B / rayligh)))
print("Target waists (m):", target_waists)
target_top_values = w0_list / target_waists
#print("Target top values:", target_top_values)
U1_batch = (np.cos(100000 * r.reshape(-1, 1)) * np.exp(-np.square(r.reshape(-1, 1) / w0_list.reshape(1, -1)))).T

plt.figure(figsize=(8, 8))
k = 2 * np.pi / wavelength
B = 0.002
xx = k * r * r[-1] / B
print("xx", xx)
kernel = j0(xx)
plt.plot(r, kernel, label="kernel")

plt.plot(r, np.square(np.abs(U1_batch[0])), color="red", label=f"before")

# for (K, abcd) in zip(K_batch, abcds):
for abcd in abcds:
    A, B, C, D = abcd
    print("------------------------------------- ABCD:", abcd)
    U2_batch = cylindrical_abcd_propagation_K(U1_batch, r, wavelength, (A, B, C, D))

    front_power1 = compute_total_power(U1_batch, r)
    front_power2 = compute_total_power(U2_batch, r)

    print("Front power before:", front_power1)
    print("Front power after:", front_power2)
    relative_error = (front_power2 - front_power1) / front_power1
    print("Relative power error per field:", relative_error)
    print("abcd:", abcd)

    # Plotting
    for i, w0 in enumerate(w0_list):
        plt.plot(r, np.square(np.abs(U2_batch[i])) + i, label=f"afterM = {(B* 1000):.3f} mm")

plt.xlabel("r (mm)")
plt.ylabel("Amplitude |U2(r)|")
plt.grid(True)
plt.legend()
plt.title("Cylindrical ABCD Propagation")
plt.tight_layout()
plt.show()