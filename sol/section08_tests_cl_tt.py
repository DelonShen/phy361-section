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
pars.AccuracyBoost = 1.0
results = camb.get_results(pars)


# ============================================================
# Test: total C_ell^TT vs CAMB
# ============================================================
camb_dl_tt = results.get_unlensed_scalar_cls(lmax=2000, CMB_unit='muK', raw_cl=False)[:, 0]
camb_l = np.arange(len(camb_dl_tt))

l_arr, Dl = cl_tt()


fig, ax = plt.subplots()
ax.plot(camb_l, camb_dl_tt, 'k',   lw=1.2, label='CAMB')
ax.plot(l_arr,  Dl,         'r--', lw=1.0, label='Ours')
ax.set_xscale('log')
ax.set_xlim(2, 1500)
ax.set_ylim(0, 5999)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\ell(\ell+1)\,C_\ell^{TT}/2\pi\ [\mu \mathrm{K}^2]$')
ax.legend(frameon=False, loc='upper left')

plt.savefig('figures/section08_cl_tt.pdf', bbox_inches='tight')
plt.close()
