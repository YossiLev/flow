import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def twisted_ribbon(n=300, length=9, width=0.2):
    t = np.linspace(-length / 2, length / 2, n)
    x = t
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    
    # Define ribbon width and twist
    w = width / 2
    
    # Compute normal and binormal vectors to twist the ribbon
    bx, by, bz = np.zeros_like(t), np.ones_like(t), np.zeros_like(t)
    
    # Define twist function with three segments
    twist = np.zeros_like(t)
    left = t < -length / 6
    middle = (t >= -length / 6) & (t <= length / 6)
    right = t > length / 6
    
    twist[middle] = np.sin(2 * np.pi * (t[middle] - t[middle][0]) / (t[middle][-1] - t[middle][0]))
    
    u = w * np.cos(np.pi * twist)
    v = w * np.sin(np.pi * twist)
    
    x1 = x
    y1 = y + u
    z1 = z + v
    
    x2 = x
    y2 = y - u
    z2 = z - v
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        np.array([x1, x2]), np.array([y1, y2]), np.array([z1, z2]),
        color='cornflowerblue', edgecolor='k'
    )
    
    ax.set_xlim([-length / 2, length / 2])
    ax.set_ylim([-width, width])
    ax.set_zlim([-width, width])
    ax.set_box_aspect([length, 1, 0.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.set_axis_off()
    plt.show()

twisted_ribbon()