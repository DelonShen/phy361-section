import numpy as np
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
    bg, cosmo, hubble, scale_factor_from_tau,
    solve_perturbations, metric_perturbations,
    IDX_DC, IDX_UC, IDX_DB, IDX_UB,
    IDX_DG, IDX_UG, IDX_PIG,
    IDX_DN, IDX_UN, IDX_PIN,
)

# grid bins
K_MIN  = 1/100 # 1/Mpc
K_MAX  = 1/0.3 # 1/Mpc
NK    = 1000
K_ARR  = np.logspace(np.log10(K_MIN), np.log10(K_MAX), NK)

ETA_MIN = 10.0
ETA_MAX = float(bg['tau0'])
N_ETA   = 10000
ETA_ARR = np.logspace(np.log10(ETA_MIN), np.log10(ETA_MAX), N_ETA)


# get perturbations
d_gamma = np.empty((NK,N_ETA))
for i, k in enumerate(K_ARR):
    sol_tca, sol_full = solve_perturbations(k)
    tau_switch = sol_tca.t[-1]
    idx_tca = (ETA_ARR < tau_switch)
    d_gamma[i, idx_tca] = sol_tca.sol(ETA_ARR[idx_tca])[IDX_DG]
    d_gamma[i, ~idx_tca] = sol_full.sol(ETA_ARR[~idx_tca])[IDX_DG]

# Hubble scale
a_of_eta = scale_factor_from_tau(ETA_ARR)
H_of_eta = np.array([hubble(a, bg) for a in a_of_eta])
kH_inv   = 1.0 / (2.0 * np.pi * a_of_eta * H_of_eta)



# Silk Damping (see Eq.(C.26) of Baumann)
from boltzmann import (
        tau_dot_of_tau, z_arr, xe_arr
)

R_0      = 3.0 * bg['rho_b'] / (4.0 * bg['rho_gamma'])
R_of_eta = R_0 * a_of_eta

_ETA_SILK = np.logspace(np.log10(1e-4), np.log10(bg['tau0']), 8000)
_a_silk   = scale_factor_from_tau(_ETA_SILK)
_R_silk   = R_0 * _a_silk
_kdot     = tau_dot_of_tau(_ETA_SILK)                        # n_e sigma_T a [1/Mpc]
_integrand = (
    1.0 / (6.0 * (1.0 + _R_silk) * _kdot)
    * (16.0/15.0 + _R_silk**2 / (1.0 + _R_silk))
)
from scipy.integrate import cumulative_trapezoid
_kS_inv_sq_cum = cumulative_trapezoid(_integrand, _ETA_SILK, initial=0.0)
kS_inv = np.sqrt(np.interp(ETA_ARR, _ETA_SILK, _kS_inv_sq_cum))

# plot photon transfer function
from matplotlib.colors import Normalize

import seaborn as sns
_cubehelix = sns.cubehelix_palette(
    start=0.5, rot=-0.75, as_cmap=True, light=0.95, dark=0.15,
    reverse=True,)

fig, ax = plt.subplots(figsize=(6,6))

ETA_grid, KINV_grid = np.meshgrid(ETA_ARR, 1/K_ARR)
pcm = ax.pcolormesh(
    ETA_grid, KINV_grid, np.abs(d_gamma),
    norm=Normalize(vmin=0.0, vmax=4.0),
    cmap=_cubehelix, shading='auto',
    rasterized=True,
)


_line_hub,  = ax.plot(ETA_ARR, kH_inv, color='k', ls='-',  lw=1.8,
                      zorder=4, label=r'Hubble scale')


_line_damp, = ax.plot(ETA_ARR, kS_inv, color='tab:red',  ls='--', lw=1.8,
                      zorder=2, label=r'Damping scale')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\eta$ [Mpc]', fontsize=14)
ax.set_ylabel(r'$k^{-1}$ [Mpc]', fontsize=14)

ax.set_xticks([10, 100, 1000])
ax.set_xticklabels(['10', '100', '1000'])
ax.set_xlim(10, 2500)

ax.set_yticks([1, 10, 100])
ax.set_ylim(None, 100)
ax.set_yticklabels(['1', '10', '100'])

ax.set_box_aspect(1.0)
ax.legend(
    handles=[_line_hub, _line_damp],
    loc='upper center', bbox_to_anchor=(0.5, -0.13),
    ncol=2, frameon=False, fontsize=11,
)
cbar = fig.colorbar(pcm, ax=ax, shrink = 1/1.618)
cbar.set_label(r'$|d_\gamma(\eta,\,k)|$',)
plt.savefig('figures/section07_photon_transfer.pdf', bbox_inches='tight', dpi=300)
