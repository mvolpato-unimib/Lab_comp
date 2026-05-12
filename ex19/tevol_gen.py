import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "text.usetex": True,           
    "font.family": "serif",        
    "font.serif": ["Palatino"],    
    "axes.labelsize": 16,          
    "xtick.labelsize": 11,         
    "ytick.labelsize": 11,         
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"
})

import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import rk4

# GENERIC PARAMETERS (CHANGEABLE)
N_p = 64
mL = 8                      # normalised lenght 
sm = 0.5                    # width of the wave packet

# TIME
d_tau = 0.005
max_t = 0.6

# PLOT PARAMETERS
plot_V_pot = True           # potential plot?
plot_psi_x = True           # mod2 along x axis
plot_psi_y = True           # mod2 along y axis (resemble a sensor)
sensor_pos = 0.7            # position of the sensor (in percentage of the generic lenght, ex. sensor_pos = 0.7 ---> sens_pos = 70% of mL)
save_anim = True           # save the animation?? 



# DERIVED PARAMETERS (FIXED)
a = np.full(2, mL / N_p)	# (m)a = (m)L / N 	phyical parameter that connects lenght to resolution
							# is equal at the right and at the left
a_norm = np.sqrt(np.sum(np.abs(a)**2))
p_max = np.pi / a			# max momentum given by NIquist theorem


# POTENTIAL
V_0 = 50

# Examples of potential to try: 
def free_part(x, y, t):
    return np.zeros_like(x)

def finite_wall(x, y, t):
    depth = a_norm * 1
    x_pos = mL/2
    return np.where((x > x_pos) & (x < x_pos + depth), V_0, 0)

def one_slit(x, y, t):
    width = a_norm * 3
    depth = a_norm * 1
    x_pos = mL/2
    y_pos = mL/2
    return np.where((x > x_pos) & (x < x_pos + depth) & 
                    ((y < y_pos - width/2) | (y > y_pos + width/2)), 
                    V_0, 0)

def one_obst(x, y, t):
    width = a_norm * 3
    depth = a_norm * 1
    x_pos = mL/2
    y_pos = mL/2
    return np.where((x > x_pos) & (x < x_pos + depth) & 
                    ((y > y_pos - width/2) & (y < y_pos + width/2)), 
                    V_0, 0)

def double_slit(x, y, t):
    width = a_norm * 2
    depth = a_norm * 1
    x_pos = mL/2
    y_pos = mL/2
    distance = a_norm * 5
    return np.where((x > x_pos) & (x < x_pos + depth) & 
                    ((y < y_pos + distance/2 - width/2) | (y > y_pos + distance/2 + width/2)) & 
                    ((y < y_pos - distance/2 - width/2) | (y > y_pos - distance/2 + width/2)), 
                    V_0, 0)

Potentials = {
    'free_part': free_part,
    'finite_wall': finite_wall,
    'one_slit': one_slit,
    'one_obst': one_obst,
    'double_slit': double_slit
}

name = 'one_obst'
V_m = Potentials[name]


# SPACE
x_0 = (mL/4, mL/2)

# MOMENTUM
p_0 = ((2*np.pi)*2, 0)

if np.sqrt(np.sum(np.abs(p_0)**2)) > np.sqrt(np.sum(np.abs(p_max)**2)):
	raise ValueError(f'p_0={p_0} > p_max={p_max}, violates Niquist theorem and results are physically inaccurate!!')

k = 1/(2 * a**2)

# COORDINATES
x_coo = np.linspace(0, mL, N_p)
y_coo = np.linspace(0, mL, N_p)
X, Y = np.meshgrid(x_coo, y_coo)

t_coo = np.arange(0,max_t, d_tau)

# -------------------------------------------

psi_0 = (np.exp(-((X - x_0[0])**2 + (Y - x_0[1])**2) / sm**2) * 
		 np.exp(1j * (p_0[0] * X + p_0[1] * Y))).flatten()
norm = np.sqrt(np.sum(np.abs(psi_0)**2))

psi_0 = (psi_0 / norm).flatten()

# Eq. Schr.
# psi(x, t+dt) = psi(x, t) - dt i H psi(x, t)
# --> d(psi) / dt = - i H psi(x, t)

# --> f_xy = - i (H psi(x, t))
def H_m (t, psi_flat):
	psi_mat = psi_flat.reshape((N_p, N_p))
	lamb = 2 * np.sum(k) + V_m(X, Y, t)

	# row shift
	row = ( k[0] * (np.roll(psi_mat, shift=-1, axis=0) + 
	np.roll(psi_mat, shift=1, axis=0)))
	
	# column shift
	col = (k[1] * (np.roll(psi_mat, shift=-1, axis=1) + 
	np.roll(psi_mat, shift=1, axis=1)))
	
	next_ij = row + col
	H_flat = (- next_ij + lamb * psi_mat).flatten()
	return -1j * H_flat

H_0 = H_m(0, psi_0)
psi_t = rk4(H_m, psi_0, t_coo)





# -------------------------------------------
# PLOTTING 


fig = plt.figure(figsize=(10, 10))
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1], hspace=0.3, wspace=0.3)

ax_2d = fig.add_subplot(gs[0, 0])
extent = [0, mL, 0, mL]
psi_mat_0 = np.abs(psi_t[0].reshape((N_p, N_p)))**2

im_psi = ax_2d.imshow(psi_mat_0, animated=True, cmap='viridis', extent=extent, origin='lower')
ax_2d.set_xlabel(r'$x$')
ax_2d.set_ylabel(r'$y$')


# Plot Potential
if plot_V_pot:
    initial_V = V_m(X, Y, t=0).astype(float)
    initial_V[initial_V == 0] = np.nan      # zeros set to nan, so that plt does not plot them and opacize the plot    
    im_V = ax_2d.imshow(initial_V, animated=True, cmap='Reds', alpha=0.5, extent=extent, origin='lower')

# Plot 1D (along x)
ax_x = None
line_x = None
if plot_psi_x:
    ax_x = fig.add_subplot(gs[1, 0])
    psi_x_0 = np.sum(psi_mat_0, axis=0)
    line_x, = ax_x.plot(x_coo, psi_x_0, color='teal', lw=2)
    max_val_x = np.max([np.sum(np.abs(p.reshape(N_p, N_p))**2, axis=0) for p in psi_t])
    
    ax_x.set_xlim(0, mL)
    ax_x.set_ylim(0, max_val_x * 1.1)
    ax_x.set_xlabel(r'$x$')
    ax_x.set_ylabel(r'$\int |\psi|^2 dy$')

# Plot 1D (along y = y1)
ax_y = None
line_y = None

# plot of the sensor
plt_psiy_ind = int(N_p * sensor_pos)
sensor = np.zeros((N_p, N_p))
sensor[:, plt_psiy_ind] =  100
sensor[sensor == 0] = np.nan        # same as for the potential, nan to make 0 == transparent
if plot_psi_y:
    ax_y = fig.add_subplot(gs[0, 1], sharey=ax_2d)
    psi_y_0 = psi_mat_0[:, plt_psiy_ind]
    line_y, = ax_y.plot(psi_y_0, y_coo, color='orangered', lw=2)
    
    max_val_y = np.max([np.abs(p.reshape(N_p, N_p)[:, plt_psiy_ind])**2 for p in psi_t])
    ax_y.set_xlim(0, max_val_y * 1.2 if max_val_y > 0 else 1)
    
    ax_y.set_ylim(0, mL)
    
    ax_y.set_xlabel(rf'$|\psi|^2$ at $x_{{det}}={x_coo[plt_psiy_ind]:.1f}$')
    plt.setp(ax_y.get_yticklabels(), visible=False)

    im_sensor = ax_2d.imshow(sensor, animated=True, cmap='Greys', alpha=0.3, extent=extent, origin='lower')

def init():
    elements = [im_psi]
    if plot_V_pot:
        elements.append(im_V)
    if plot_psi_x:
        line_x.set_ydata(np.sum(psi_mat_0, axis=0))
        elements.append(line_x)
    if plot_psi_y:
        line_y.set_xdata(psi_mat_0[:, plt_psiy_ind]) 
        elements.append(line_y)
        elements.append(im_sensor)

    return elements

def update(frame):
    psi_mat = np.abs(psi_t[frame].reshape((N_p, N_p)))**2
    psi2_t = np.abs(psi_t[frame].reshape((N_p, N_p)))**2
    im_psi.set_array(psi_mat)
    
    elements = [im_psi]
    
    if plot_V_pot:
        v_data = V_m(X, Y, t_coo[frame]).astype(float)
        v_data[v_data == 0] = np.nan
        im_V.set_array(v_data)
        elements.append(im_V)
        
    if plot_psi_x:
        psi_x = np.sum(psi_mat, axis=0)
        line_x.set_ydata(psi_x)
        elements.append(line_x)
        
    if plot_psi_y:
        psi_y_slice = psi2_t[:, plt_psiy_ind]
        line_y.set_xdata(psi_y_slice) 
        elements.append(line_y)

        s_data = sensor.copy() 
        im_sensor.set_array(s_data)
        elements.append(im_sensor)
    
    return elements

ani = FuncAnimation(
    fig, 
    update, 
    frames=len(psi_t), 
    init_func=init, 
    blit=True, 
    interval=20, 
    repeat=True
)

if save_anim:
    name_folder = 'plots/' + name
    try:
        print("Saving the animation...")
        ani.save(f'{name_folder}.gif', writer='pillow', fps=15)
        print(f'Animation saved in "{name_folder}.gif"')
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")

else:
    plt.show()
