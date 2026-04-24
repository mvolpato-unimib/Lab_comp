import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection 
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
#         PLOT
# --------------------------

fig, ax = plt.subplots()

lc = LineCollection([], cmap='plasma', lw=2)
lc.set_norm(plt.Normalize(-1.2, 1.2))
ax.add_collection(lc)
# ----------------------------------------------------------------

time_text = ax.text(0.75, 0.9, '', transform=ax.transAxes, fontsize=12, fontweight='bold')
ax.set_xlim(0, Lmax)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x,t)$')

def update(frame):
    t = times[frame]
    u_pt = udf_0 * np.exp(-alpha * ps**2 * t)

    u_t = np.real(heat_dft.idft(u_pt)) 
    
    points = np.array([xcoo, u_t]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc.set_segments(segments)
    lc.set_array(u_t[:-1]) 

    current_time = frame * t_max * 0.01
    time_text.set_text(f'Tempo: {current_time:.2f}s')

    return lc, time_text


ani = FuncAnimation(fig, update, frames=len(times), interval=50)

plt.tight_layout()
ani.save('plots/ani_col.gif', writer='pillow')
ani.save('plots/ani_col.mp4', writer='ffmpeg')
plt.show()