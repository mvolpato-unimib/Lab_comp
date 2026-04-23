import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
sys.path.append(os.path.abspath('..'))

from lib_equations import DFT
# import lib_plot

# params
Lmax = 2*np.pi
N_p = 128
alpha = 0.4
t_max = 1


xcoo = np.linspace(0, Lmax, N_p, endpoint=True, dtype=float)
times = np.linspace(0, t_max, 100, dtype=float)
u0 = np.sin(xcoo) + 0.5 * np.sin(3*xcoo)
heat_dft = DFT(xcoo)
ps = heat_dft.p

# fourier transform of u(x,0) (p-space)
udf_0 = heat_dft.dft(u0)


# --------------------------
#           PLOT
# --------------------------

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
time_text = ax.text(0.75, 0.9, '', transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
ax.set_xlim(0, Lmax)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x,t)$')

def update(frame):
    t = times[frame]
    # u(p,t), analitical solution in the p-space
    u_pt = udf_0 * np.exp(-alpha * ps**2 * t)

    # inverse fourier transform: u(p,t) ---> u(x,t) (x-space)
    u_t = heat_dft.idft(u_pt)
    line.set_data(xcoo, u_t)
    current_time = frame * t_max *0.01
    time_text.set_text(f'Tempo: {current_time:.2f}s')

    return line, time_text

ani = FuncAnimation(fig, update, frames=len(times), interval=50)

plt.tight_layout()
ani.save('ani_blue.gif', writer='pillow')
ani.save('ani_blue.mp4', writer='ffmpeg')
plt.show()