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
)
results = camb.get_results(pars)


# ============================================================
# Test 1: Peebles C-factor
#   compare against Baumann Fig. 3.16
# ============================================================
my_C = np.array([peebles_rhs(zi, xe_arr[i], bg, return_C=True)
                 for i, zi in enumerate(z_arr)])

plt.figure()
plt.plot(z_arr, my_C, 'k')
plt.xlabel('$z$')
plt.ylabel('Peebles $C$-factor')
plt.yscale('log')
plt.ylim(9e-4, 2)
plt.xlim(0, 2000)
plt.xticks([0, 500, 1000, 1500, 2000])
plt.savefig('figures/section02_peebles_C.pdf', bbox_inches='tight')
plt.close()




# ============================================================
# Test 2: Ionization history x_e(z)
# ============================================================
z = np.logspace(np.log10(30), np.log10(1800), 500)

CAMB_xe = results.get_background_redshift_evolution(
    z, ['x_e'], format='dict',
)['x_e']
my_xe = np.interp(z, z_arr[::-1], xe_arr[::-1])
my_Saha_xe = saha_xe(cosmo['T_cmb'] * (1 + z),
                     (1 - cosmo['Y_He']) * bg['rho_b'] / m_H * (1 + z)**3)

plt.figure()

plt.plot(z, my_Saha_xe, '--', c='lightgrey', label='Saha')
plt.plot(z, CAMB_xe, 'k', label='CAMB')
plt.plot(z, my_xe, 'r--', label='Ours')


plt.xlabel('$z$')
plt.xlim(30, 1800)
plt.ylabel('$x_e(z)$')

plt.yscale('log')
plt.ylim(1e-4, 2)
plt.legend(frameon=False)
plt.savefig('figures/section02_xe.pdf', bbox_inches='tight')
plt.close()





# ============================================================
# Test 3: Baryon temperature
# ============================================================
CAMB_Tb = results.get_background_redshift_evolution(
    z, ['T_b'], format='dict',
)['T_b']
my_Tb = cosmo['T_cmb'] * (1 + z)

plt.figure()
plt.plot(z, CAMB_Tb, 'k', label='CAMB')
plt.plot(z, my_Tb, 'r--', label='$T_b(z) = T_{\gamma}(z)$')
plt.axvline(200, color='gray', lw=0.7, ls='--')
plt.xlabel('$z$')
plt.xlim(30, 1800)

plt.ylabel('$T_b(z)$ [K]')
plt.loglog()
plt.legend(frameon=False, loc='lower right')
plt.savefig('figures/section02_Tb.pdf', bbox_inches='tight')
plt.close()


plt.plot(z, (CAMB_Tb-my_Tb)/CAMB_Tb, 'r', label = '$T_b(z)$ Rel. Err.')
plt.plot(z, (CAMB_xe-my_xe)/CAMB_xe, 'b', label='$x_e(z)$ Rel. Err.')
plt.axvline(200, color='gray', lw=0.7, ls='--')

plt.legend(frameon=False)
plt.xlabel('$z$')

plt.ylim(-0.2, 0.2)
plt.xlim(200, 1800)
plt.savefig('figures/section02_residual.pdf', bbox_inches='tight')

