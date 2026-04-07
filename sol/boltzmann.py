"""
Units: distances and times in Mpc (with c = 1), H in 1/Mpc
       densities in kg/m^3, temperatures in K.

Assuming:
    - Spatially flat
    - Massless neutrinos
"""

import numpy as np
import scipy


# ============================================================
# PHYSICAL CONSTANTS (SI)
# ============================================================

c_SI = 2.99792458e8              # speed of light [m/s]
c_km_s = c_SI / 1e3              # speed of light [km/s]
k_B = 1.380649e-23               # Boltzmann constant [J/K]
h_P = 6.62607015e-34             # Planck constant [J·s]
m_e = 9.1093837015e-31           # electron mass [kg]
G = 6.67430e-11                  # gravitational constant [m^3/kg/s^2]
Mpc_in_m = 3.0856775814913673e22 # 1 Mpc in metres
sigma_SB = 5.670374419e-8        # Stefan-Boltzmann constant [W/m^2/K^4]



# Critical density for h = 1: rho_crit,100 = 3(100 km/s/Mpc)^2 /(8 pi G)
_H100_SI = 100.0 * 1e3 / Mpc_in_m
_rho_crit_100 = 3.0 * _H100_SI**2 / (8.0 * np.pi * G)


# ============================================================
# DEFAULT COSMOLOGICAL PARAMETERS (Planck 2018 best-fit ΛCDM)
# see 1807.06209 Table 1
# ============================================================

cosmo = {
    'Omega_b_h2': 0.02238,       # Omega_b h^2 - baryon physical density
    'Omega_c_h2': 0.1201,        # Omega_c h^2 - CDM physical density
    'h': 0.6732,                 # H_0 / (100 km/s/Mpc)
    'n_s': 0.96605,              # scalar spectral index
    'A_s': 2.1e-9,               # scalar amplitude at k_pivot
    'tau_reion': 0.0544,         # reionization optical depth
    'N_eff': 3.044,              # effective number of massless neutrino species
    'T_cmb': 2.7255,             # CMB temperature today [K]
    'Y_He': 0.2454,              # helium mass fraction by weight
}



# ============================================================
# BACKGROUND COSMOLOGY
# ============================================================
 
bg = {}


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
