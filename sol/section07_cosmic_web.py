import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family": "serif",
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# our Boltzmann code
from boltzmann import (
    bg, cosmo, hubble,
    solve_perturbations, metric_perturbations,
    IDX_DC, IDX_UC, IDX_DB, IDX_UB,
    IDX_DG, IDX_UG, IDX_PIG,
    IDX_DN, IDX_UN, IDX_PIN,
)

# Box and grid
LBOX = 512.0 # Mpc
NGRID = 128
dX = LBOX/NGRID

k_F = 2 * np.pi / LBOX
K_NYQ = np.pi * NGRID / LBOX

# Get P(k, z=0), this is what we did in `section07_linear_matter_power.py`
K_MIN = 0.5 * k_F
K_MAX = 1.8 * K_NYQ
NK    = 50
K_ARR = np.logspace(np.log10(K_MIN), np.log10(K_MAX), NK)


A_s   = cosmo['A_s']
n_s   = cosmo['n_s']
k_piv = 0.05  # 1/Mpc, CAMB default

rho_c = bg['rho_c']
rho_b = bg['rho_b']

P_k = np.empty_like(K_ARR)
for i, k in enumerate(K_ARR):
    _, sol_full = solve_perturbations(k)
    y0 = sol_full.y[:, -1]
    d_c = y0[IDX_DC]; u_c = y0[IDX_UC]
    d_b = y0[IDX_DB]; u_b = y0[IDX_UB]
    d_g = y0[IDX_DG]; u_g = y0[IDX_UG]; pi_g = y0[IDX_PIG]
    d_n = y0[IDX_DN]; u_n = y0[IDX_UN]; pi_n = y0[IDX_PIN]

    a0  = 1.0
    aH0 = a0 * hubble(a0, bg)
    _, Psi = metric_perturbations(
        d_g, u_g, pi_g, d_n, u_n, pi_n,
        d_c, u_c, d_b, u_b, a0, aH0, k, bg,
    )
    delta_c_N = d_c + 3.0 * Psi
    delta_b_N = d_b + 3.0 * Psi
    delta_m_N = (rho_c * delta_c_N + rho_b * delta_b_N) / (rho_c + rho_b)

    Delta2_R = A_s * (k / k_piv)**(n_s - 1.0)
    P_k[i] = (2.0 * np.pi**2 / k**3) * Delta2_R * delta_m_N**2


# interpolate in log-log
from scipy.interpolate import interp1d

f_logPk = interp1d(
    np.log(K_ARR), np.log(P_k),
    kind='linear', fill_value='extrapolate',
)


# Generate white noise field
_k = 2.0 * np.pi * np.fft.fftfreq(NGRID, d=dX)          # 1/Mpc
kmesh = np.stack(np.meshgrid(_k, _k, _k, indexing='ij'))   # (3, N, N, N)
k_abs = np.sqrt(_k[:, None, None]**2 + _k[None, :, None]**2 + _k[None, None, :]**2)


delta_q_0 = np.random.normal(size=(NGRID, NGRID, NGRID))
delta_q_0 /= np.sqrt(dX**3)

delta_k_0 = np.fft.fftn(delta_q_0) * dX**3
delta_k_0 *= np.sqrt(np.exp(f_logPk(np.log(k_abs))))
delta_k_0[0, 0, 0] = 0.0 # Kill DC mode

delta_q_0 = np.real(np.fft.ifftn(delta_k_0)) / dX**3

# Zeldovich displacement
# I'll make the matter dominated approximation D ~ a here for simplicity

SCALEFACTOR = 1
s1_k = 1j * kmesh / k_abs**2 * delta_k_0
s1_k[:, 0, 0, 0] = 0.0


s1_q = np.real(np.fft.ifftn(s1_k, axes=(-3, -2, -1))) / dX**3
q = np.indices((NGRID, NGRID, NGRID)) * dX
ZA_x = q + SCALEFACTOR * s1_q

# simplest way to visualize is with scatter plots
fig = plt.figure(figsize=(6.0, 6.0), dpi=400)
ax = fig.add_subplot(projection='3d')
ax.scatter(
    ZA_x[0], ZA_x[1], ZA_x[2],
    marker=',', alpha=0.04*6, s=0.05, lw=0, c='k',
)

ax.set_xlim(-20, LBOX + 20)
ax.set_ylim(-20, LBOX + 20)
ax.set_zlim(-20, LBOX + 20)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_axis_off()
ax.set_title(r'Zeldovich Approx, $z = 0$')
ax.set_aspect('equal', adjustable='box')

plt.savefig('figures/section07_cosmic_web_simple.png', bbox_inches='tight',)

# there's a more "correct" way to visualize/estimate the densities
# below is a closer approximation to the "correct" way than what we do above
# see (Abel+12 1111.3944) or talk to Tom

def get_tet_centroids(Ndim, p3d):
    """Centroids of all tetrahedra spanned by neighbouring grid points.

    p3d : (N, N, N, 3) Lagrangian-indexed particle positions.
    Returns (Np*6, 3) of centroid positions.
    """
    vert = np.array((
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ))
    conn = np.array((
        (4, 0, 7, 1), (1, 0, 7, 3), (5, 1, 4, 7),
        (2, 3, 1, 7), (1, 5, 6, 7), (2, 6, 7, 1),
    ))
    Ntetpp = len(conn)
    Np = Ndim**3
    cen = np.zeros((Np * Ntetpp, 3))
    for m in range(Ntetpp):
        off = vert[conn[m]]
        sl = lambda v: (slice(v[0], Ndim + v[0]),
                        slice(v[1], Ndim + v[1]),
                        slice(v[2], Ndim + v[2]))
        orig = p3d[sl(off[3]) + (slice(None),)]
        a = (p3d[sl(off[0]) + (slice(None),)] - orig).reshape((Np, 3))
        b = (p3d[sl(off[1]) + (slice(None),)] - orig).reshape((Np, 3))
        c = (p3d[sl(off[2]) + (slice(None),)] - orig).reshape((Np, 3))
        cen[m::Ntetpp, :] = orig.reshape(Np, 3) + (a + b + c) / 4.0
    return cen

p3d = np.moveaxis(ZA_x, 0, -1) # (N, N, N, 3)
p3d = p3d[:, :, :, ::-1] # xyz -> zyx (C/Python -> Fortran index convention basically)
ZA_cen = get_tet_centroids(NGRID - 1, p3d)

fig = plt.figure(figsize=(6.0, 6.0), dpi=400)
ax = fig.add_subplot(projection='3d')

ax.scatter(
    ZA_cen[:, 2], ZA_cen[:, 1], ZA_cen[:, 0],
    marker=',', alpha=0.04, s=0.05, lw=0, c='k',
)

ax.set_xlim(-20, LBOX + 20)
ax.set_ylim(-20, LBOX + 20)
ax.set_zlim(-20, LBOX + 20)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_axis_off()
ax.set_title(r'Zeldovich Approx, $z = 0$')
ax.set_aspect('equal', adjustable='box')

plt.savefig('figures/section07_cosmic_web_fancy.png', bbox_inches='tight',)
