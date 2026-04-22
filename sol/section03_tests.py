import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family" : "serif",
    'figure.figsize': (5, 5/1.618),
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
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
# Test 1: Reionization x_e(z)
# ============================================================
z = np.linspace(0, 30, 500)

CAMB_xe = results.get_background_redshift_evolution(
    z, ['x_e'], format='dict',
)['x_e']
my_xe = np.interp(z, z_arr[::-1], xe_arr[::-1])

plt.figure()
plt.plot(z, CAMB_xe, 'k', label='CAMB')
plt.plot(z, my_xe, 'r--', label='Ours')
plt.xlabel('$z$')
plt.ylabel('$x_e(z)$')
plt.xlim(0, 30)
plt.legend(frameon=False)
plt.savefig('figures/section03_xe_reion.pdf', bbox_inches='tight')
plt.close()




# ============================================================
# Test 2: Optical depth tau(z) over full range
# ============================================================
z = np.logspace(-2, np.log10(np.max(z_arr)), 500)

# optical_depth expects z decreasing (eta increasing)
z_desc = z[::-1]
a_desc = 1.0 / (1.0 + z_desc)
eta_desc = conformal_time(a_desc, bg)

xe_desc = np.interp(z_desc, z_arr[::-1], xe_arr[::-1])
my_kappa_dot = thomson_opacity(z_desc, xe_desc, bg)
my_tau = optical_depth(eta_desc, my_kappa_dot)

CAMB_kappa_dot = results.get_background_redshift_evolution(
    z_desc, ['opacity'], format='dict',
)['opacity']
CAMB_tau = optical_depth(eta_desc, CAMB_kappa_dot)

plt.figure()
plt.axhline(cosmo['tau_reion'], color='lightgrey', ls='-', lw=1,
            label=r'$\tau_{\rm reion.}$')
plt.plot(z, CAMB_tau[::-1], 'k', label='CAMB')
plt.plot(z, my_tau[::-1], 'r--', label='Ours')
plt.xlabel('$z$')
plt.xlim(1e0, 2000)
plt.ylim(2e-3,None)
plt.ylabel(r'$\tau(z)$')
plt.loglog()
plt.legend(frameon=False)
plt.savefig('figures/section03_tau.pdf', bbox_inches='tight')
plt.close()




# ============================================================
# Test 3: Visibility g(eta)
# should compare with Fig.7.5 of Baumann
# ============================================================
z = np.logspace(-3, np.log10(np.max(z_arr)), 10000)
z = np.hstack(([0], z))

z_desc = z[::-1]
a_desc = 1.0 / (1.0 + z_desc)
eta_desc = conformal_time(a_desc, bg)

xe_desc = np.interp(z_desc, z_arr[::-1], xe_arr[::-1])
my_kappa_dot = thomson_opacity(z_desc, xe_desc, bg)
my_tau = optical_depth(eta_desc, my_kappa_dot)
my_g = my_kappa_dot * np.exp(-my_tau)

CAMB_g = results.get_background_redshift_evolution(
    z_desc, ['visibility'], format='dict',
)['visibility']

# split into recombination (z > z_split) and reionization (z < z_split) bumps
# so the reionization bump can be rescaled to be visible alongside recombination
z_split = 30.0
recomb_mask = z_desc > z_split
reion_mask = z_desc <= z_split
reion_scale = 200.0

fig, ax = plt.subplots()
# recombination bump at natural amplitude
ax.plot(eta_desc[recomb_mask], CAMB_g[recomb_mask], 'k', label='CAMB')
ax.plot(eta_desc[recomb_mask], my_g[recomb_mask], 'r--', label='Ours')
# reionization bump, rescaled by reion_scale
ax.plot(eta_desc[reion_mask], reion_scale * CAMB_g[reion_mask], 'k')
ax.plot(eta_desc[reion_mask], reion_scale * my_g[reion_mask], 'r--')


# label each bump at its peak
i_recomb = np.argmax(my_g[recomb_mask])
ax.text(eta_desc[recomb_mask][i_recomb],
        my_g[recomb_mask][i_recomb] * 1.02,
        'Recombination', ha='center', va='bottom', fontsize=11)

i_reion = np.argmax(my_g[reion_mask])
ax.text(eta_desc[reion_mask][i_reion],
        reion_scale * my_g[reion_mask][i_reion] * 1.05,
        r'Reionization ($\times %d$)'%reion_scale, ha='center', va='bottom', fontsize=11)

ax.set_xscale('log')
ax.set_xlim(70, np.max(eta_desc))
ax.set_xlabel(r'$\eta$ [Mpc]')
ax.set_xticks([100, 1e3, 1e4], ['100', '1000', '10000'])
ax.set_ylabel(r'$g(\eta)$ [Mpc$^{-1}$] ')
ax.set_ylim(None, 0.025)
ax.legend(frameon=False)


# for adding z-axis to the top
def eta_to_z(e):
    return np.interp(e, eta_desc, z_desc)

def z_to_eta(zz):
    return np.interp(zz, z_desc[::-1], eta_desc[::-1])

secax = ax.secondary_xaxis('top', functions=(eta_to_z, z_to_eta))
secax.set_xlabel('$z$')
secax.set_xticks([1000, 100, 10, 1], ['1000', '100', '10', '1'])
secax.minorticks_off()

plt.savefig('figures/section03_visibility.pdf', bbox_inches='tight')
plt.close()
