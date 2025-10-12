# plotting the Feynman propagator D_F on the 2D plane f = (t, x, 0, 0)
# We use the closed-form expression with Bessel K:
# D_F(f) = m/(4*pi^2 * sqrt(x^2 - t^2 + i eps)) * K1(m * sqrt(x^2 - t^2 + i eps))
# where f^2 = t^2 - x^2, and we set m=1 for units.
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
import math

# parameters and grid
m = 1.0
tmin, tmax, nt = -5.0, 5.0, 501
xmin, xmax, nx = -5.0, 5.0, 501
t = np.linspace(tmin, tmax, nt)
x = np.linspace(xmin, xmax, nx)
T, X = np.meshgrid(t, x, indexing='xy')  # T along horizontal axis in plots will be t, vertical axis x

# compute complex sqrt argument with + i*eps to implement Feynman prescription
eps = 1e-6
arg = X**2 - T**2 + 1j*eps  # equals (x^2 - t^2) + i eps
# complex sqrt; choose principal branch via numpy.sqrt for complex numbers
Z = np.sqrt(arg)

# avoid division by zero at Z=0 by adding tiny offset
Z_safe = Z.copy()
Z_safe[np.abs(Z_safe) == 0] = 1e-12 + 0j

# compute D_F
prefactor = m / (4.0 * math.pi**2 * Z_safe)
D = prefactor * kv(1, m * Z_safe)

# compute absolute value and phase
absD = np.log(np.abs(D))
phaseD = np.angle(D)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))  # 2x2 grid of subplots
# Plot 1: absolute value of D_F
alt = axes[0, 0]
alt.imshow(absD.T, origin='lower', extent=[tmin, tmax, xmin, xmax], aspect='auto')
alt.set_xlabel('f_t (t)')
alt.set_ylabel('f_x (x)')
alt.set_title(r'Absolute value |D_F(f)|, m=1 (f = (t,x,0,0))')
##alt.figure.colorbar(label='|D_F|')

# Plot 2: phase of D_F
alt = axes[0, 1]
alt.imshow(phaseD.T, origin='lower', extent=[tmin, tmax, xmin, xmax], aspect='auto')
alt.set_xlabel('f_t (t)')
alt.set_ylabel('f_x (x)')
alt.set_title(r'Phase arg(D_F(f)), m=1 (f = (t,x,0,0))')
#alt.set_colorbar(label='phase (radians)')

# Plot 3: absolute value along x=0 slice
alt = axes[1, 0]
di = absD[300,:]
alt.plot(di.T)
alt.set_xlabel('f_t (t)')
alt.set_title(r'Absolute value |D_F(f)|, m=1 (f = (50,x,0,0))')

# Plot 4: absolute value along x=10 slice
alt = axes[1, 1]
alt.plot(np.diag(absD, 10).T)
alt.set_xlabel('f_t (t)')
alt.set_title(r'Absolute value |D_F(f)|, m=1 (f = (x,x + 10,0,0))')

plt.tight_layout()
plt.show()
