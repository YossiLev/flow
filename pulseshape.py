import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters ---
T = 10e-12           # Total time window [s]
Nt = 512             # Number of time points
dt = T / Nt          # Time step
t = np.linspace(-T/2, T/2, Nt)

Nz = Nt              # Spatial grid (same as time points for split-step)
dz = 1e-6            # Spatial step [m]
L_total = Nz * dz    # Total cavity length [m]

# Time for one round trip
c = 3e8              # Speed of light [m/s]
n_eff = 3.5          # Effective refractive index
round_trip_time = L_total * n_eff / c

# --- Gain and absorber sections ---
gain_mask = np.zeros(Nt, dtype=bool)
abs_mask = np.zeros(Nt, dtype=bool)
gain_mask[:int(0.6 * Nt)] = True
abs_mask[int(0.6 * Nt):] = True

# --- Initial fields and carriers ---
E = np.random.normal(0, 1e-8, Nt).astype(np.complex128)  # Initial noise
N_g = np.full(Nt, 1.5e24)  # Gain carrier density [1/m^3]
N_a = np.full(Nt, 1.0e24)  # Absorber carrier density [1/m^3]

# --- Constants ---
g0 = 2e-20       # Gain coefficient [m^2]
a0 = 1e-20       # Absorption coefficient [m^2]
N_tr = 1e24      # Transparency carrier density [1/m^3]
N_sat = 1.5e24   # Saturation density for absorber [1/m^3]
gamma = 500      # Total losses per round trip [1/s]

# Carrier dynamics
tau_g = 1e-9     # Gain carrier lifetime [s]
tau_a = 0.5e-9   # Absorber carrier lifetime [s]
I_g = 10e-3      # Injection current [A]
q = 1.6e-19      # Elementary charge
V_g = 1e-18      # Active region volume [m^3]
Gamma_g = 1.0    # Gain confinement factor
Gamma_a = 1.0    # Absorber confinement factor

# Frequency axis for dispersion (optional)
f = np.fft.fftfreq(Nt, d=dt)
omega = 2 * np.pi * f
beta2 = 0        # Group velocity dispersion [s^2/m] (can be nonzero)

# --- Simulation loop ---
n_round_trips = 200
pulse_history = []

# Re-seed initial field with a stronger Gaussian noise to trigger instability
np.random.seed(42)
E = (1e-6 * (np.random.randn(Nt) + 1j * np.random.randn(Nt))).astype(np.complex128)

for rt in range(n_round_trips):
    I = np.abs(E)**2

    # Carrier update
    g_term = g0 * np.maximum(N_g - N_tr, 0)
    a_term = a0 * (1 - N_a / N_sat)

    dNg = (I_g / (q * V_g) - N_g / tau_g - Gamma_g * g_term * I) * dt
    dNa = (-N_a / tau_a - Gamma_a * a_term * I) * dt

    N_g[gain_mask] += dNg[gain_mask]
    N_a[abs_mask] += dNa[abs_mask]

    # Gain and absorption
    g = g0 * np.maximum(N_g - N_tr, 0)
    a = a0 * (1 - N_a / N_sat)

    # Nonlinear gain/loss step
    G_total = g - a - gamma
    E *= np.exp(0.5 * G_total * round_trip_time)

    # Linear dispersion step (optional, set beta2 = 0 if off)
    E_fft = np.fft.fft(E)
    E_fft *= np.exp(-0.5j * beta2 * omega**2 * round_trip_time)
    E = np.fft.ifft(E_fft)

    # Save history
    if rt % 10 == 0 or rt > n_round_trips - 10:
        pulse_history.append(np.abs(E)**2)

    # Optional: print pulse energy to monitor growth
    if rt % 20 == 0:
        energy = np.sum(np.abs(E)**2) * dt
        print(f"Round trip {rt:3d}: Pulse energy = {energy:.2e} J")


# --- Plot pulse evolution ---
t_ps = t * 1e12  # convert to picoseconds
pulse_history = np.array(pulse_history)

plt.figure(figsize=(10, 5))
plt.imshow(
    pulse_history,
    extent=[t_ps[0], t_ps[-1], 0, pulse_history.shape[0]],
    aspect='auto',
    origin='lower',
    cmap='inferno'
)
plt.colorbar(label='Pulse Intensity |E(t)|²')
plt.xlabel('Time (ps)')
plt.ylabel('Stored Frame (every 10th + last 10)')
plt.title('Pulse Shaping Over Round Trips')
plt.tight_layout()
plt.show()
