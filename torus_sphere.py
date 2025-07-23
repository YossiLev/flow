import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plt.tight_layout()

# Parametrize the sphere and torus
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 60)
u, v = np.meshgrid(u, v)

def sphere():
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)
    return x, y, z

def torus(R=1.5, r=0.5):
    u2 = u
    v2 = 2 * v  # scale v to match torus param
    x = (R + r * np.cos(v2)) * np.cos(u2)
    y = (R + r * np.cos(v2)) * np.sin(u2)
    z = r * np.sin(v2)
    return x, y, z

xs, ys, zs = sphere()
xt, yt, zt = torus()

def update(frame):
    ax.clear()
    t = frame / num_frames
    w = np.sin(np.pi * t)**2  # smooth blending

    x = (1 - w) * xs + w * xt
    y = (1 - w) * ys + w * yt
    z = (1 - w) * zs + w * zt

    ax.plot_surface(x, y, z, color='plum', alpha=0.8, edgecolor='gray', linewidth=0.1)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_axis_off()
    ax.set_title(f"Sphere ↔ Torus Morph (t = {t:.2f})", fontsize=14)

# Animation
num_frames = 60
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

plt.show()
