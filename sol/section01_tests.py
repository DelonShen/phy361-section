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
# Test 1: Hubble parameter H(z)
# ============================================================


z = np.logspace(-1, 4, 500)
a = 1.0 / (1.0 + z)


CAMB_hubble = results.hubble_parameter(z)
my_hubble = hubble(a, bg) * c_km_s

plt.figure()
plt.plot(z, CAMB_hubble, 'k', label='CAMB')
plt.plot(z, my_hubble, 'r--', label='Mine')
plt.xlabel('z')
plt.ylabel('$H(z)$')
plt.loglog()
plt.savefig('figures/section01_hubble.pdf', bbox_inches='tight')
plt.close()


plt.plot(z, (CAMB_hubble-my_hubble)/CAMB_hubble, 'k')
plt.xlabel('z')
plt.ylabel('$H(z)$ Rel. Err.')
plt.xscale('log')
plt.show()





# ============================================================
# Test 2: Conformal Time tau(z)
# ============================================================

CAMB_tau = results.conformal_time(z)
my_tau = conformal_time(a, bg)

plt.figure()
plt.plot(z, CAMB_tau, 'k', label='CAMB')
plt.plot(z, my_tau, 'r--', label='Mine')
plt.xlabel('z')
plt.ylabel(r'$\tau(z)$')
plt.loglog()
plt.savefig('figures/section01_conformal_time.pdf', bbox_inches='tight')
plt.close()


plt.plot(z, (CAMB_tau-my_tau)/CAMB_tau, 'k')
plt.xlabel('z')
plt.ylabel(r'$\tau(z)$ Rel. Err.')
plt.xscale('log')
plt.show()




# ============================================================
# Test 3: Density parameters Omega_i(z)
# ============================================================

my_Omega = density_fractions(a, bg)
CAMB_Omega = {
    name: results.get_Omega(name, z=z)
    for name in ['photon', 'neutrino', 'cdm', 'baryon', 'de']
}

labels = {
    'photon': r'$\Omega_\gamma$', 'neutrino': r'$\Omega_\nu$',
    'cdm': r'$\Omega_c$', 'baryon': r'$\Omega_b$', 'de': r'$\Omega_\Lambda$',
}

plt.figure()
for i, name in enumerate(labels):
    plt.plot(z, CAMB_Omega[name]-my_Omega[name], label=labels[name])
plt.xlabel('z')
plt.ylabel(r'$\Omega_i(z)$ Rel. Err.')

plt.legend(frameon=False)

plt.xscale('log')
plt.savefig('figures/section01_density_parameter.pdf', bbox_inches='tight')
plt.close()
