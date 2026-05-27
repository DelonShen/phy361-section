import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family" : "serif",
    'figure.figsize': (5, 5/1.618),
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

import camb

from boltzmann import *




pars = camb.set_params(
    H0=100 * cosmo['h'], ombh2=cosmo['Omega_b_h2'],
    omch2=cosmo['Omega_c_h2'], ns=cosmo['n_s'], As=cosmo['A_s'],
    tau=cosmo['tau_reion'], TCMB=cosmo['T_cmb'],
    nnu=cosmo['N_eff'], YHe=cosmo['Y_He'], mnu=0,
    lmax=2000,
)
pars.Want_CMB_lensing = False
pars.AccuracyBoost = 2.0
results = camb.get_results(pars)


# ============================================================
# Test: Delta_ell(k) for a single ell
# ============================================================

L_TARGET = 2


# CAMB transfer functions
ct = results.get_cmb_transfer_data('scalar')
camb_L = np.asarray(ct.L, dtype=int)
camb_q = np.asarray(ct.q, dtype=float)
camb_Dlk = np.asarray(ct.delta_p_l_k[0, :, :])
li = int(np.argmin(np.abs(camb_L - L_TARGET)))
l_used = int(camb_L[li])


# Use the same grids as `cl_tt` (k_source, default_tau_grid)
k_source, _ = default_k_grids(bg)
taus = default_tau_grid(bg)

ours = np.zeros(len(k_source))
for ik, k in enumerate(k_source):
    S, _ = los_source_grid(k, taus)
    ours[ik] = delta_l_k(np.array([l_used]), k, S, taus)[0]


fig, ax = plt.subplots()
ax.plot(camb_q,   camb_Dlk[li, :], 'k',   lw=1.2, label='CAMB')
ax.plot(k_source, ours,            'r--', lw=1.0, label='Ours')

ax.set_xlim(1e-5, 4.2e-2)
plt.xscale('log')
ax.set_xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')
ax.set_ylabel(r'$\Delta_\ell(k)$')
ax.text(0.05, 0.95, r'$\ell = %d$' % l_used,
        transform=ax.transAxes, ha='left', va='top')
ax.legend(frameon=False, loc='upper right')

plt.savefig('figures/section08_delta_l_k.pdf', bbox_inches='tight')
plt.close()
