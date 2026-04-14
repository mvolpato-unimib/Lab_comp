import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import bisec

e = 0.5
a = 1
M_t = lambda t: 0.5 * t

# M(T) = 0.5*T = 2*pi - e*sin(2*pi)    ----->    0.5T - 2pi = 0
find_T = lambda t: M_t(t) - 2 * np.pi

# Research of the period for a given E_t, eccentric anomaly
T = bisec(find_T, 0, 20)

# Evaluate E_t
t_span = np.linspace(0, T, 500)
M_ts = np.array([M_t(t) for t in t_span])

E_ls = []
for M_ti in M_ts:
    find_Et = lambda E_t: E_t - e * np.sin(E_t) - M_ti
    E_ls.append(bisec(find_Et, 0, 20))
E_ts = np.array(E_ls)

find_xt = lambda E_ti: a * (np.cos(E_ti) - e)
find_yt = lambda E_ti: a * np.sqrt(1 - e**2) * np.sin(E_ti)
x_ts = np.array([find_xt(E_ti) for E_ti in E_ts])
y_ts = np.array([find_yt(E_ti) for E_ti in E_ts])



# animation

n_points_back = 30  
fig, ax = plt.subplots(figsize=(8, 8))
plt.xlim(-1.8, 1.0) 
plt.ylim(-1.2, 1.2) 
ax.set_aspect('equal')

area_patch = ax.fill([], [], color="#FF8F6D", alpha=0.3, edgecolor='none', label=f'Area spazzata in [p-{n_points_back}, p]')[0]
orbit, = ax.plot(x_ts, y_ts, 'k--', alpha=0.3, label='Orbita')
sun, = ax.plot(0, 0, '*', markersize=30, label='Sole (Fuoco)', color='gold', markeredgecolor='orange')
planet, = ax.plot([], [], 'ro', markersize=10, label='Pianeta')
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

def init():
    planet.set_data([], [])
    area_patch.set_xy(np.empty((0, 2)))
    time_text.set_text('')
    return planet, area_patch, time_text


def update(frame):
    """
    frame: indice dell'array x_ts, y_ts
    """
    # index is always positive
    start_idx = max(0, frame - n_points_back)
    
    # position of the planet at the moment
    xp = x_ts[frame]
    yp = y_ts[frame]
    planet.set_data([xp], [yp])
    
    # Update of the polygon 
    # polygon == Sun -> curve -> Sun
    x_arch = x_ts[start_idx : frame + 1]
    y_arch = y_ts[start_idx : frame + 1]
    
    x_vertex = np.concatenate(([0], x_arch, [0]))
    y_vertex = np.concatenate(([0], y_arch, [0]))
    
    area_patch.set_xy(np.column_stack([x_vertex, y_vertex]))
    
    current_t = t_span[frame]
    time_text.set_text(f'Tempo: {current_t:.2f}')
    
    return planet, area_patch, time_text

ani = FuncAnimation(fig, update, frames=len(t_span),
                    init_func=init, blit=True, interval=5, repeat=True)

plt.legend(
    loc='upper right', 
    markerscale=0.5
)

# print("Salvataggio in corso...")
# try:
#     # Usiamo fps=30 e riduciamo il carico se necessario
#     ani.save('plots/anim_orbita.gif', writer='pillow', fps=30)
#     print("Salvataggio completato!")
# except Exception as e:
#     print(f"Errore durante il salvataggio: {e}")

plt.show()