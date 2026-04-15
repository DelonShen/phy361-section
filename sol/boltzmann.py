"""
Units: distances and times in Mpc (with c = 1), H in 1/Mpc
       densities in kg/m^3, temperatures in K.

Assuming:
    - Spatially flat
    - Massless neutrinos
"""

import numpy as np
import scipy
from constants import *



# ============================================================
# BACKGROUND COSMOLOGY
# ============================================================
 
bg = {}


# Critical density for h = 1: rho_crit,100 = 3(100 km/s/Mpc)^2 /(8 pi G)
_H100_SI = 100.0 * 1e3 / Mpc_in_m
_rho_crit_100 = 3.0 * _H100_SI**2 / (8.0 * np.pi * G)



# densities at present day

# radiation energy density = 4 sigma T^4 / c
# dividing by c^2 gives equivalent mass density
bg['rho_gamma'] = (4.0 * sigma_SB / c_SI * cosmo['T_cmb']**4) / c_SI**2

# matter
bg['rho_c'] = cosmo['Omega_c_h2'] * _rho_crit_100
bg['rho_b'] = cosmo['Omega_b_h2'] * _rho_crit_100


bg['rho_nu'] = cosmo['N_eff'] * (7.0 / 8.0) * (4.0 / 11.0)**(4.0 / 3.0) * bg['rho_gamma']


# DE, assuming flat universe
H0_SI = 100.0 * cosmo['h'] * 1e3 / Mpc_in_m
rho_crit = 3.0 * H0_SI**2 / (8.0 * np.pi * G)
bg['rho_Lambda'] = rho_crit - bg['rho_gamma'] - bg['rho_c'] - bg['rho_b'] - bg['rho_nu']


def rho_total(a, bg):
    #Total energy density at scale factor a [kg/m^3].
    return ((bg['rho_gamma'] + bg['rho_nu']) / a**4
            + (bg['rho_c'] + bg['rho_b']) / a**3
            + bg['rho_Lambda'])

def hubble(a, bg):
    #Hubble parameter H(a) = sqrt(8 pi G / 3 * rho_total) in 1/Mpc."""
    return np.sqrt((8.0 * np.pi * G / 3.0) * rho_total(a, bg) * (Mpc_in_m / c_SI)**2 )

def dtauda(a, bg):
    #Conformal time integrand d tau/d a = 1/(a^2 H) in Mpc.
    return 1.0 / (a**2 * hubble(a, bg))

def conformal_time(a, bg):
    #Conformal time tau(a) = \int_0^a da'/(a'**2 * H(a')) in Mpc
    a = np.atleast_1d(np.asarray(a, dtype=float))
    result = np.array([
        scipy.integrate.quad(dtauda, 0, ai, args=(bg,), epsabs = 0.0, epsrel=1e-10)[0]
        for ai in a
    ])
    return result.squeeze()

def density_fractions(a, bg):
    #Fractional density Omega_i(a) = rho_i(a)/rho_total(a) for each species."""
    rho_tot = rho_total(a, bg)
    return {
        'photon':   bg['rho_gamma'] / a**4 / rho_tot,
        'neutrino': bg['rho_nu']    / a**4 / rho_tot,
        'cdm':      bg['rho_c']     / a**3 / rho_tot,
        'baryon':   bg['rho_b']     / a**3 / rho_tot,
        'de':       bg['rho_Lambda']        / rho_tot,
    }

bg['tau0'] = conformal_time(1.0, bg)
bg['a_eq'] = ((bg['rho_gamma'] + bg['rho_nu'])/ (bg['rho_c'] + bg['rho_b']))
bg['tau_eq'] = conformal_time(bg['a_eq'], bg)


# ============================================================
# RECOMBINATION
#   Peebles 3-level atom
#   Assumes T_b(z) = T_CMB(z) [no baryon temperature evolution]
#   Ignores helium recombination
# ============================================================

# $(CR \times T)^{3/2} = \left(\frac{m_e k_B T}{2\pi\hbar^2}\right)^{3/2}$
CR = 2.0 * np.pi * m_e * k_B / h_P**2  # Saha prefactor [m^-2 K^-1]
 
def saha_xe(T, n_H):
    #Hydrogen Saha equilibrium x_e, `recombination.pdf` Eq. (1)
    #solving for xe with positive root of quadratic equation
    s = (CR * T)**1.5 * np.exp(-E_I / T) / n_H
    return (-s + np.sqrt(s * s + 4.0 * s)) / 2.0

def peebles_rhs(z, x_e, bg, return_C = False):
    #Peebles ODE RHS of dx_e / dz, `recombination.pdf` Eq. (3)

    a = 1.0 / (1.0 + z)
    T = cosmo['T_cmb'] * (1.0 + z)
    n_H = (1 - cosmo['Y_He']) * bg['rho_b'] / m_H * (1.0 + z)**3

    H_SI = hubble(a, bg) * c_SI / Mpc_in_m  # Mpc^-1 -> s^-1

    # Case-B recombination coefficient (Pequignot, Petitjean & Boisson 1991)
    t4 = T / 1e4
    alpha_B = a_PPB * t4**b_PPB / (1.0 + c_PPB * t4**d_PPB)

    # Photoionization rate from n=2, by detailed balance:
    beta_B = 0.25 * (CR * T)**1.5 * np.exp(-E_2_bind / T) * alpha_B

    # Sobolev Lyman-alpha escape rate
    #   R_Lya = 8 pi H / (3 n_{1s} lambda_Lya^3)
    n_1s = (1.0 - x_e) * n_H
    R_Lya = 8.0 * np.pi * H_SI / (3.0 * n_1s * lambda_Lya**3)

    numer = 0.75 * R_Lya + 0.25 * Lambda_2s1s

    #Peebles C-factor, `recombination.pdf` Eq. (2)
    C = numer / (beta_B + numer)

    if(return_C):
        # return the Peebles C-factor
        # instead of the RHS of the ODE
        return C

    recomb = n_H * x_e**2 * alpha_B
    photoion = 4.0 * (1.0 - x_e) * beta_B * np.exp(-E_21 / T)
    return C / (H_SI * (1.0 + z)) * (recomb - photoion)

# z-sampling for recombination history
nz = 10000
z_arr = np.linspace(3000, 0, nz)
xe_arr = np.empty(nz)

# find redshift where x_e first drops below 0.99
# to set IC for Peebles ODE
IC_idx = None

for i, z in enumerate(z_arr):
    T = cosmo['T_cmb'] * (1.0 + z)
    n_H = (1 - cosmo['Y_He']) * bg['rho_b'] / m_H * (1.0 + z)**3
    xe_arr[i] = saha_xe(T, n_H)
    if(xe_arr[i] < 0.99):
        IC_idx = i
        break


# solve Peebles ODE after finding IC
z_ode = z_arr[IC_idx:]
sol = scipy.integrate.solve_ivp(
    peebles_rhs, [z_ode[0], z_ode[-1]], [xe_arr[IC_idx]],
    t_eval=z_ode, method='LSODA',
    rtol=1e-8, atol=0.0,
    args=(bg,)
)

n_sol = min(sol.y.shape[1], len(z_ode))
xe_arr[IC_idx:IC_idx + n_sol] = sol.y[0, :n_sol]
