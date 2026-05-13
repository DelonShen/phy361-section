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
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 14,
})

import camb
from camb.symbolic import Delta_g, Delta_c, Delta_b, Delta_r, q_g, Phi_N

from boltzmann import *




pars = camb.set_params(
    H0=100 * cosmo['h'], ombh2=cosmo['Omega_b_h2'],
    omch2=cosmo['Omega_c_h2'], ns=cosmo['n_s'], As=cosmo['A_s'],
    tau=cosmo['tau_reion'], TCMB=cosmo['T_cmb'],
    nnu=cosmo['N_eff'], YHe=cosmo['Y_He'], mnu=0,
)
results = camb.get_results(pars)


# CAMB returns Newtonian-gauge delta_a and Phi_N (= our Psi). Convert to BS03 d_a:
#   d_a = delta_a / (1 + w_a) - 3 Phi_N
#   u_g = (3/4) q_g / k
def camb_perts(k, tau):
    out = results.get_time_evolution(
        k, tau,
        vars=[Delta_g, q_g, Delta_c, Delta_b, Delta_r, Phi_N],
        frame='Newtonian',
    )
    delta_g, q_g_c, delta_c, delta_b, delta_r, phi_n = out.T
    return {
        'd_g': 0.75 * delta_g - 3.0 * phi_n,
        'u_g': 0.75 * q_g_c / k,
        'd_c': delta_c       - 3.0 * phi_n,
        'd_b': delta_b       - 3.0 * phi_n,
        'd_n': 0.75 * delta_r - 3.0 * phi_n,
    }


def our_perts(sol_tca, sol_full, tau):
    # piecewise: TCA before the switch, full system after.
    tau_switch = sol_full.t[0]
    y = np.empty((NSTATE, len(tau)))
    m = tau < tau_switch
    if np.any(m):
        y[:, m] = sol_tca.sol(tau[m])
    if np.any(~m):
        y[:, ~m] = sol_full.sol(tau[~m])
    return {
        'd_g': y[IDX_DG],
        'u_g': y[IDX_UG],
        'd_c': y[IDX_DC],
        'd_b': y[IDX_DB],
        'd_n': y[IDX_DN],
    }


KEYS   = ['d_g', 'u_g', 'd_c', 'd_b', 'd_n']
LABELS = [r'$d_\gamma$', r'$u_\gamma$ [Mpc]',
          r'$d_c$', r'$d_b$', r'$d_\nu$']


# ============================================================
# More detailed comparison: 3 k modes x 5 perturbation variables
# ============================================================
K_LIST = [0.01, 0.1, 0.5]

fig, axes = plt.subplots(len(K_LIST), len(KEYS),
                         figsize=(14, 7), sharex='row')

for irow, k in enumerate(K_LIST):
    sol_tca, sol_full = solve_perturbations(k)
    tau = np.logspace(np.log10(sol_tca.t[0] * 1.05),
                      np.log10(bg['tau0'] * 0.999), 600)
    ours   = our_perts(sol_tca, sol_full, tau)
    camb_d = camb_perts(k, tau)

    for icol, (key, label) in enumerate(zip(KEYS, LABELS)):
        ax = axes[irow, icol]
        ax.plot(tau, camb_d[key], 'k',   lw=1.2, label='CAMB')
        ax.plot(tau, ours[key],   'r--', lw=1.0, label='Ours')
        ax.set_xscale('log')
        ax.set_xlim(tau[0], tau[-1])
        if key == 'd_c':
            ax.set_ylim(0, 15)
        elif key == 'd_b':
            ax.set_ylim(-2.3, 4.3)
        if irow == 0:
            ax.set_title(label)
        if icol == 0:
            ax.set_ylabel(r'$k = %.2f\ {\rm Mpc}^{-1}$' % k)
        if irow == len(K_LIST) - 1:
            ax.set_xlabel(r'$\tau$ [Mpc]')

axes[0, 0].legend(frameon=False, fontsize=10, loc='lower left')

plt.savefig('figures/section06_expansive.pdf', bbox_inches='tight')
plt.close()
