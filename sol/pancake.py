import numpy as np


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

N_p     = int(1e5)+1   # number of particles
N_grid  = int(1e4)+1   # number of grid cells
a_init  = 0.1          # init scale factor
a_cross = 0.5          # when shell cross
a_final = 1.7          # last snapshot
da      = 1e-4         # time step


# ============================================================
# DENSITY ASSIGNMENT (Cloud-in-Cell)
# ============================================================


def CIC_deposit(x, N_grid):
    N_grid = int(N_grid)
    dx = 1.0 / N_grid
    m  = 1.0 / len(x)

    left = x - 0.5 * dx
    xi   = np.int64(left / dx)
    frac = 1.0 + xi - left / dx

    # wrap the particles whose left edge fell below x=0
    ind        = np.where(left < 0.0)
    frac[ind]  = -(left[ind] / dx)
    xi[ind]    = N_grid - 1

    xir             = xi + 1
    xir[xir==N_grid] = 0

    rho  = np.bincount(xi,  weights=frac       * m, minlength=N_grid)
    rho += np.bincount(xir, weights=(1.-frac)  * m, minlength=N_grid)
    return rho * N_grid  # multiply by 1/dx to make rho a density


def CIC_interpolate(x, grad_phi):
    N_grid = len(grad_phi)
    dx    = 1.0 / N_grid
    left  = x - 0.5 * dx
    xi    = np.int64(left / dx)
    frac  = 1.0 + xi - left / dx

    # wrap particles whose left edge fell below x=0
    ind        = np.where(left < 0.0)
    frac[ind]  = -(left[ind] / dx)
    xi[ind]    = N_grid - 1

    xir              = xi + 1
    xir[xir==N_grid] = 0
    return frac * grad_phi[xi] + (1.0 - frac) * grad_phi[xir]




# ============================================================
# FFT POISSON SOLVER
#   See lecture notes for derivation of Green's function
# ============================================================

def solve_poisson(rho, a):
    N_grid   = len(rho)
    delta_l = np.fft.rfft(rho - 1.0)
    k_l     = 2.0 * np.pi * np.fft.rfftfreq(N_grid)

    phi_l       = np.zeros_like(k_l, dtype=np.complex128)
    phi_l[1:]   = -3.0 / (8.0 * a * N_grid**2) * delta_l[1:] / np.sin(k_l[1:] / 2.0)**2
    phi_l[0]    = 0.0

    return np.fft.irfft(phi_l)




# ============================================================
# ACCELERATION AT PARTICLE POSITIONS
#   Central differences for grad(phi) at cell centers, then CIC
#   interpolation back to particles. 
# ============================================================


def array_periodic_boundary(x):
    # enforce x in [0, 1)
    x[x >= 1.0] -= 1.0
    x[x <  0.0] += 1.0
    return x


def central_difference(y):
    return (np.roll(y, -1) - np.roll(y, 1)) / 2.0



# ============================================================
# LEAPFROG (kick-drift-kick)
# ============================================================

def kick(x, p, a, da, N_grid):
    rho      = CIC_deposit(x, N_grid)
    phi      = solve_poisson(rho, a)
    grad_phi = central_difference(phi) * N_grid
    ap       = CIC_interpolate(x, grad_phi)
    return p - np.sqrt(a) * da * ap


def drift(x, p, a, da):
    x = x + a**(-1.5) * da * p
    return array_periodic_boundary(x)


# ============================================================
# ZELDOVICH PANCAKE INITIAL CONDITIONS
# ============================================================


def zeldovich_x(q, a, A):
    return q + a * A * np.sin(2.0 * np.pi * q)

def zeldovich_p(q, a, A):
    return a**1.5 * A * np.sin(2.0 * np.pi * q)


q = np.linspace(0.0, 1.0, N_p)
x = zeldovich_x(q, a_init, 1 / (a_cross * 2.0 * np.pi))
p = zeldovich_p(q, a_init, 1 / (a_cross * 2.0 * np.pi))



# ============================================================
# EVOLVE
# ============================================================

from tqdm import trange # for progress bar

N_steps = int((a_final - a_init) / da)
a_cur   = a_init

snapshots = []

for i in trange(N_steps):
    if(i%10 == 0):
        snapshots.append({'a': a_cur, 'x': x.copy(), 'p': p.copy()})

    # kick - drift - kick
    p      = kick( x, p, a_cur,        da/2, N_grid)
    x      = drift(x, p, a_cur + da/2, da          )
    p      = kick( x, p, a_cur + da,   da/2, N_grid)
    a_cur += da

snapshots.append({'a': a_cur, 'x': x.copy(), 'p': p.copy()})


# ============================================================
# BONUS (two-stream instability)
#   What we've built here is also basically 
#   a very basic particle-in-cell (PIC) Plasma code
#   lets look a the classic two stream instability
# ============================================================

def solve_poisson_plasma(rho):
    # resuse our `solve_poisson`, a=-1.5 flips sign and makes force repulsive
    return solve_poisson(rho, a=-1.5)


# remove scale factors from KDK 
def kick_plasma(x, v, dt, N_grid):
    rho      = CIC_deposit(x, N_grid)
    phi      = solve_poisson_plasma(rho)
    grad_phi = central_difference(phi) * N_grid
    ap       = CIC_interpolate(x, grad_phi)
    return v - dt * ap

def drift_plasma(x, v, dt):
    x = x + dt * v
    return array_periodic_boundary(x)


v_stream  = 0.05
v_pert    = 1e-2
dt        = 1e-3
t_final   = 50.0

q_p = (np.arange(N_p / 10) + 0.5) / (N_p / 10)

# set up two streams of electrons
x = np.concatenate([q_p, q_p])
v = np.concatenate([
    +v_stream + v_pert * np.sin(2.0 * np.pi * q_p),
    -v_stream - v_pert * np.sin(2.0 * np.pi * q_p),
])



N_steps   = int(t_final / dt)
t_cur     = 0.0
plasma_snapshots = []

for i in trange(N_steps):
    if(i%10 == 0):
        plasma_snapshots.append({'t': t_cur, 'x': x.copy(), 'v': v.copy()})
    v      = kick_plasma (x, v, dt/2, N_grid)
    x      = drift_plasma(x, v, dt)
    v      = kick_plasma (x, v, dt/2, N_grid)
    t_cur += dt

plasma_snapshots.append({'t': t_cur, 'x': x.copy(), 'v': v.copy()})
