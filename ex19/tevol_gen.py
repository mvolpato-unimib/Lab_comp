import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

import sys
import os
sys.path.append(os.path.abspath('..'))
from lib_equations import rk4

# GENERIC PARAMETERS (CHANGEABLE)
N_p = 128
mL = 8                      # normalised lenght 
sm = 0.5                    # width of the wave packet
Dir_cond = True             # Dirichlet conditions (the wf is set to 0 at the borders, resulting in a reflection)

# SPACE
x_0 = (mL/4, mL/2)                  # initial position of the gaussian wave packet                      

# MOMENTUM
p_0 = ((2*np.pi)*6 / mL, 0)         # dimensionless momentum

# TIME
d_tau = 0.002
max_t = 0.6

# PLOT PARAMETERS
plot_V_pot = True           # potential plot?
plot_psi_x = True           # mod2 along x axis
plot_psi_y = True           # mod2 along y axis at a specific x=sensor_pos (resemble a sensor)
sensor_pos = 0.7            # position of the sensor (in percentage of the generic lenght, ex. sensor_pos = 0.7 ---> sens_pos = 70% of mL)
plot_time = False
save_anim = False            # save the animation?? 
if save_anim:
    plt.rcParams.update({
        "text.usetex": True,           
        "font.family": "serif",        
        "font.serif": ["Palatino"],    
        "axes.labelsize": 16,          
        "xtick.labelsize": 11,         
        "ytick.labelsize": 11,         
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"
    })



# DERIVED PARAMETERS (FIXED)
a = np.full(2, mL / N_p)	# (m)a = (m)L / N 	phyical parameter that connects lenght to resolution
							# is equal at the right and at the left
a_norm = np.sqrt(np.sum(np.abs(a)**2))
a_x = np.sqrt(np.abs(a[0])**2)
a_y = np.sqrt(np.abs(a[1])**2)

# d_tau < (a^2) / 2         # approximation for the CFL limit 
cfl_limit = (min(a)**2) / 2

if d_tau > cfl_limit:
    print(f"d_tau ({d_tau}) > CFL limit ({cfl_limit:.6f})!")
    print("It's possible to occur in overflow... Automatic reduction")
    
    d_tau = cfl_limit * 0.8  # 20% margin from the limit
    print(f'd_tau set to {d_tau}')



# POTENTIAL
V_0 = 100       # near inifinite wall potential wall
unit_len = 2 * a_norm
depth = unit_len * 1/2
width = unit_len * 3

x_pos = mL/2
y_pos = mL/2

# Examples of potential to try: 
def free_part(x, y, t):
    return np.zeros_like(x)

def finite_wall(x, y, t):
    V_tunnel = 20
    return np.where((x >= x_pos) & (x <= x_pos + depth), V_tunnel, 0)

def one_slit(x, y, t):
    return np.where((x >= x_pos) & (x <= x_pos + depth) & 
                    ((y <= y_pos - width/2) | (y >= y_pos + width/2)), 
                    V_0, 0)

def one_obst(x, y, t):
    return np.where((x >= x_pos) & (x <= x_pos + depth) & 
                    ((y >= y_pos - width/2) & (y <= y_pos + width/2)), 
                    V_0, 0)

def double_slit(x, y, t):
    distance = a_norm*4
    return np.where((x >= x_pos) & (x <= x_pos + depth) & 
                    ((y <= y_pos + distance - width/2) | (y >= y_pos + distance + width/2)) & 
                    ((y <= y_pos - distance - width/2) | (y >= y_pos - distance + width/2)), 
                    V_0, 0)

def alpha_p(x, y, t):
    return np.where((x <= x_pos), 0, 10 + 1/(x+1e-10))


Potentials = {
    'free_part': free_part,
    'finite_wall': finite_wall,
    'one_slit': one_slit,
    'one_obst': one_obst,
    'double_slit': double_slit,
    'alpha_p': alpha_p
}

name = 'alpha_p'
V_m = Potentials[name]


# Niquist check
p_max_x = np.pi / a_x                   # max (dimensionless) momentum given by NIquist theorem
p_max_y = np.pi / a_y         

if np.abs(p_0[0]) > np.abs(p_max_x):
	raise ValueError(f'p_0={np.abs(p_0[0])} > p_max={np.abs(p_max_x)}, violates Niquist theorem and results are physically inaccurate!!')
if np.abs(p_0[1]) > np.abs(p_max_y):
	raise ValueError(f'p_0={np.abs(p_0[1])} > p_max={np.abs(p_max_y)}, violates Niquist theorem and results are physically inaccurate!!')

k = 1/(2 * a**2)

# COORDINATES
x_coo = np.linspace(0, mL, N_p, endpoint=False)     # excluding the last point we assure that the number of intervals is N_p
y_coo = np.linspace(0, mL, N_p, endpoint=False)
X, Y = np.meshgrid(x_coo, y_coo)

t_coo = np.arange(0,max_t, d_tau)

# -------------------------------------------

psi_0 = (np.exp(-((X - x_0[0])**2 + (Y - x_0[1])**2) / sm**2) * 
		 np.exp(1j * (p_0[0] * X + p_0[1] * Y))).flatten()

area_element = a[0] * a[1]

def eval_norm2(psi):
    return np.sum(np.abs(psi)**2) * area_element

norm = np.sqrt(eval_norm2(psi_0))
an_norm = (2 / (np.pi * sm**2))**(-1/2)

# if needed it is possible to compare the two definitions of norm
#   - numerical definition
#   - analitical definition

# print('\nnorm =', norm)
# print('analitic norm =', an_norm)
# print()

psi_0 = psi_0 / norm




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
    
    if Dir_cond:
        row[0, :] = 0; row[-1, :] = 0  # Dirichlet conditions
        col[:, 0] = 0; col[:, -1] = 0  # DC
	
    next_ij = row + col
    H_flat = (- next_ij + lamb * psi_mat).flatten()
    return -1j * H_flat

H_0 = H_m(0, psi_0)

psi_t_complex = rk4(H_m, psi_0, t_coo)
psi_t = np.abs(psi_t_complex)**2   

# in case of prove of the contant constant mod2 of the wf, after the normalisation
# print('|psi_0|^2 =', eval_norm2(psi_0))
# norm_t = np.sum(psi_t, axis=1) * area_element
# print('|psi_t|^2 =\n', norm_t[::int(len(norm_t)/10)])




# -------------------------------------------
# PLOTTING 


fig = plt.figure(figsize=(10, 10))
gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1], hspace=0.3, wspace=0.3)

ax_2d = fig.add_subplot(gs[0, 0])
extent = [0, mL, 0, mL]
psi_mat_0 = psi_t[0].reshape((N_p, N_p))

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
    psi_x_0 = np.sum(psi_mat_0, axis=0)* a[1]
    line_x, = ax_x.plot(x_coo, psi_x_0, color='teal', lw=2)
    max_val_x = np.max([np.sum(p.reshape(N_p, N_p) , axis=0)* a[1] for p in psi_t])
    
    ax_x.set_xlim(0, mL)
    ax_x.set_ylim(0, max_val_x * 1.1)
    ax_x.set_xlabel(r'$x$')
    ax_x.set_ylabel(r'$\int |\psi|^2 dy$')
    if plot_time:
        stats_text = ax_x.text(0.75, 0.90, '', transform=ax_x.transAxes,
                           fontsize=12, fontweight='bold', va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
    
    max_val_y = np.max([p.reshape(N_p, N_p)[:, plt_psiy_ind]  for p in psi_t])
    ax_y.set_xlim(0, max_val_y * 1.2 if max_val_y > 0 else 1)
    
    ax_y.set_ylim(0, mL)
    
    ax_y.set_xlabel(rf'$|\psi|^2$, $x_{{det}}={x_coo[plt_psiy_ind]:.1f}$')
    ax_y.tick_params(axis='y', labelleft=False)

    im_sensor = ax_2d.imshow(sensor, animated=True, cmap='Greys', alpha=0.3, extent=extent, origin='lower')

def init():
    elements = [im_psi]
    if plot_V_pot:
        elements.append(im_V)
    if plot_psi_x:
        line_x.set_ydata(np.sum(psi_mat_0, axis=0))
        if plot_time:
            stats_text.set_text('') 
        elements.append(line_x)
        if plot_time:
            elements.append(stats_text)
    if plot_psi_y:
        line_y.set_xdata(psi_mat_0[:, plt_psiy_ind]) 
        elements.append(line_y)
        elements.append(im_sensor)

    return elements

def update(frame):
    psi_mat = psi_t[frame].reshape((N_p, N_p))
    im_psi.set_array(psi_mat)
    
    elements = [im_psi]
    
    if plot_V_pot:
        v_data = V_m(X, Y, t_coo[frame]).astype(float)
        v_data[v_data == 0] = np.nan
        im_V.set_array(v_data)
        elements.append(im_V)
        
    if plot_psi_x:
        psi_x = np.sum(psi_mat, axis=0)* a[1]
        line_x.set_ydata(psi_x)
        if plot_time:
            current_t = t_coo[frame]
            stats_text.set_text(rf'Time: {current_t:.3f}')        
            elements.append(stats_text)
        elements.append(line_x)
        
    if plot_psi_y:
        psi_y_slice = psi_mat[:, plt_psiy_ind]
        line_y.set_xdata(psi_y_slice) 
        elements.append(line_y)
    
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
    print("Saving the animation...")
    ani.save(f'{name_folder}.gif', writer='pillow', fps=15)
    print(f'Animation saved in "{name_folder}.gif"')

else:
    plt.show()
