"""
Constants used in `boltzmann.py`
"""

# ============================================================
# PHYSICAL CONSTANTS (SI)
#   source: Particle data book
# ============================================================

c_SI = 2.99792458e8              # speed of light [m/s]
c_km_s = c_SI / 1e3              # speed of light [km/s]
k_B = 1.380649e-23               # Boltzmann constant [J/K]
h_P = 6.62607015e-34             # Planck constant [J·s]
m_e = 9.1093837015e-31           # electron mass [kg]
G = 6.67430e-11                  # gravitational constant [m^3/kg/s^2]
Mpc_in_m = 3.0856775814913673e22 # 1 Mpc in metres
sigma_SB = 5.670374419e-8        # Stefan-Boltzmann constant [W/m^2/K^4]


# ============================================================
# DEFAULT COSMOLOGICAL PARAMETERS (Planck 2018 best-fit ΛCDM)
#   see 1807.06209 Table 1
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
# RECOMBINATION CONSTANTS 
# ============================================================
m_H = 1.673575e-27               # hydrogen atom mass [kg]

Lambda_2s1s = 8.2206             # two-photon decay rate 2s -> 1s [s^-1] (Labzowsky et al 2005)

_eV_to_J = 1.602176634e-19       # eV to Joules conversion
E_I = 13.605698 * _eV_to_J / k_B # H ionization energy [K]
E_21 = 3.0 * E_I / 4.0           # Lyman-alpha energy E_2 - E_1 = (3/4) E_I [K]
E_2_bind = E_I / 4.0             # n=2 binding energy |E_2| = E_I/4 [K]


# Lyman-alpha wavelength [m]: lambda_Lya = h c / (k_B E_21) -- derived from E_21
lambda_Lya = h_P * c_SI / (k_B * E_21)

# Case-B recombination coefficient (Pequignot, Petitjean & Boisson 1991):
#   alpha_B(T) = a_PPB t4^b_PPB / (1 + c_PPB t4^d_PPB),  t4 = T / 1e4
a_PPB = 4.309e-19
b_PPB = -0.6166
c_PPB = 0.6703
d_PPB = 0.5300
