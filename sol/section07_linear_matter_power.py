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
    bg, cosmo, hubble,
    solve_perturbations, metric_perturbations,
    IDX_DC, IDX_UC, IDX_DB, IDX_UB,
    IDX_DG, IDX_UG, IDX_PIG,
    IDX_DN, IDX_UN, IDX_PIN,
)


# CAMB
import camb
from camb.symbolic import Delta_c, Delta_b

pars = camb.set_params(
    H0=100 * cosmo['h'], ombh2=cosmo['Omega_b_h2'],
    omch2=cosmo['Omega_c_h2'], ns=cosmo['n_s'], As=cosmo['A_s'],
    tau=cosmo['tau_reion'], TCMB=cosmo['T_cmb'],
    nnu=cosmo['N_eff'], YHe=cosmo['Y_He'], mnu=0,
)
pars.InitPower.set_params(As=cosmo['A_s'], ns=cosmo['n_s'], pivot_scalar=0.05)
pars.set_matter_power(redshifts=[0.0], kmax=2.0)
results = camb.get_results(pars)




K_MIN = 3e-4   # 1/Mpc
K_MAX = 0.2    # 1/Mpc
NK    = 100
K_ARR = np.logspace(np.log10(K_MIN), np.log10(K_MAX), NK)

# get petrubations on K_ARR
def our_delta_cb(k):
    _, sol_full = solve_perturbations(k)
    y0 = sol_full.y[:, -1]

    d_c = y0[IDX_DC]; u_c = y0[IDX_UC]
    d_b = y0[IDX_DB]; u_b = y0[IDX_UB]
    d_g = y0[IDX_DG]; u_g = y0[IDX_UG]; pi_g = y0[IDX_PIG]
    d_n = y0[IDX_DN]; u_n = y0[IDX_UN]; pi_n = y0[IDX_PIN]

    a0  = 1.0
    aH0 = a0 * hubble(a0, bg)  # conformal H today

    Phi, Psi = metric_perturbations(
        d_g, u_g, pi_g, d_n, u_n, pi_n,
        d_c, u_c, d_b, u_b,
        a0, aH0, k, bg,
    )

    # Newtonian gauge
    delta_c_N = d_c + 3.0 * Psi
    delta_b_N = d_b + 3.0 * Psi
    # Synchronous gauge
    delta_c_S = delta_c_N + 3.0 * aH0 * u_c
    delta_b_S = delta_b_N + 3.0 * aH0 * u_c
    return delta_c_N, delta_b_N, delta_c_S, delta_b_S



delta_c_N_ours = np.empty(NK)
delta_b_N_ours = np.empty(NK)

delta_c_S_ours = np.empty(NK)
delta_b_S_ours = np.empty(NK)

for i, k in enumerate(K_ARR):
    delta_c_N_ours[i], delta_b_N_ours[i], delta_c_S_ours[i], delta_b_S_ours[i] = our_delta_cb(k)


# get total matter
w_c = bg['rho_c'] / (bg['rho_c'] + bg['rho_b'])
w_b = bg['rho_b'] / (bg['rho_c'] + bg['rho_b'])
delta_m_N_ours = w_c * delta_c_N_ours + w_b * delta_b_N_ours
delta_m_S_ours = w_c * delta_c_S_ours + w_b * delta_b_S_ours

# get Newtonain-gauge power spectrum from CAMB
# CAMB default spits out Synchronous
camb_evo = results.get_time_evolution(
    K_ARR, np.array([bg['tau0']]),
    vars=[Delta_c, Delta_b], frame='Newtonian',
)
delta_c_camb_N = camb_evo[:, 0, 0]
delta_b_camb_N = camb_evo[:, 0, 1]
delta_m_camb_N = w_c * delta_c_camb_N + w_b * delta_b_camb_N


# compute P(k)
# note typically A_s is the comoving curvature power at pivot k=0.05
Delta2_R = cosmo['A_s'] * (K_ARR / 0.05)**(cosmo['n_s'] - 1.0)

# Delta2_R = k^3/2pi^2 P_R
P_ours_N = (2.0 * np.pi**2 / K_ARR**3) * Delta2_R * delta_m_N_ours**2
P_ours_S = (2.0 * np.pi**2 / K_ARR**3) * Delta2_R * delta_m_S_ours**2
P_camb_N = (2.0 * np.pi**2 / K_ARR**3) * Delta2_R * delta_m_camb_N**2

# get CAMB's synchronous gauge power spectrum
k_camb, _, pk_camb = results.get_linear_matter_power_spectrum(
    var1='delta_nonu', var2='delta_nonu',
    hubble_units=False, k_hunit=False,
    have_power_spectra=True,
)

P_camb_S = np.interp(K_ARR, k_camb, pk_camb[0])


# plot everything
plt.figure(figsize=(6,6/1.618))

plt.plot(K_ARR, P_camb_N, c='#00ffff', lw=2.6, label=r'\texttt{CAMB} Newtonian')
plt.plot(K_ARR, P_camb_S, c='#ff00ff', lw=2.6, label=r'\texttt{CAMB} Synchronous')
plt.plot(K_ARR, P_ours_N, 'k--',  lw=1.0, label=r'Ours')
plt.plot(K_ARR, P_ours_S, 'k--',  lw=1.0)

plt.ylabel(r'$P(k,\,z=0)\ [\mathrm{Mpc}^{3}]$')
plt.xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')

plt.legend(frameon=False)
plt.xscale('log')
plt.yscale('log')

plt.xlim(K_ARR[0], K_ARR[-1])

plt.savefig('figures/section07_linear_matter_power.pdf', bbox_inches='tight')
