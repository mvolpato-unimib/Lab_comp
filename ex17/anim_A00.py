import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import rk4

# parameters
A = 0
gammas = [0.3, 0.6, 0.9]
y0 = np.array([np.pi/4, 0.0])
t_span = np.linspace(0, 15, 200)
L = 1.0



# ------------------------------------


def damp_osc(t, u, gamma_val):
    u0, u1 = u
    du0 = u1
    du1 = -np.sin(u0) - gamma_val * u1 + A * np.sin(2/3 * t)
    return np.array([du0, du1])

trajectories = []
theta_data = []
for g in gammas:
    sol = rk4(lambda t, u: damp_osc(t, u, g), y0, t_span)
    theta = sol[:, 0]
    trajectories.append((L * np.sin(theta), -L * np.cos(theta)))
    theta_data.append(theta)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
pendulum_axes = axes[0, :]
plot_axes = axes[1, :]

lines = []
time_texts = []
points = []

for i, ax in enumerate(pendulum_axes):
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(-L, L)
    ax.set_ylim(-L-0.1, 0.1)
    ax.set_aspect('equal')
    ax.set_title(rf'$A=0,\ \gamma = {gammas[i]}$')
    line, = ax.plot([], [], 'o-', lw=2, color='navy')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    lines.append(line)
    time_texts.append(time_text)

for i, ax in enumerate(plot_axes):
    ax.plot(t_span, theta_data[i], color='gray', alpha=0.5)
    point, = ax.plot([], [], 'ro')
    ax.set_xlim(t_span[0], t_span[-1])
    ax.set_ylim(np.min(theta_data)-0.1*np.max(theta_data), np.max(theta_data)+0.1*np.max(theta_data))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\theta(t)$')
    points.append(point)

def init():
    for line, time_text in zip(lines, time_texts):
        line.set_data([], [])
        time_text.set_text('')
    for point in points:
        point.set_data([], [])
    return lines + time_texts + points

def animate(i):
    for j in range(3):
        x_data, y_data = trajectories[j]
        lines[j].set_data([0, x_data[i]], [0, y_data[i]])
        time_texts[j].set_text(f't = {t_span[i]:.1f}s')
        points[j].set_data([t_span[i]], [theta_data[j][i]])
    return lines + time_texts + points

ani = FuncAnimation(fig, animate, frames=len(t_span),
                    interval=50, blit=True, init_func=init)

plt.tight_layout()

ani.save('anim/damp_pend_A00.mp4', writer='ffmpeg', fps=30)
plt.show()