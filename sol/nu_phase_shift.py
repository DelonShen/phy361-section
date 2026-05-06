import numpy as np
from scipy.integrate import solve_ivp
from constants import *



# integration range
TAU0, TAU_END = 2e-6, 12

# we will solve for a single fourier mode
K = 2 * np.pi 

# ============================================================
# BACKGROUND COSMOLOGY
#   will be similar to `boltzmann.py` (`background.pdf`)
#   but only track photons and nu (Omega_r,0 = 1)
#   with density ratio R_nu
# ============================================================

bg = {}

# Critical density for h = 1: rho_crit,100 = 3(100 km/s/Mpc)^2 /(8 pi G)
# for simplicity (lazyness) I'll just assume h = 1 in this program
_H100_SI = 100.0 * 1e3 / Mpc_in_m
_rho_crit_100 = 3.0 * _H100_SI**2 / (8.0 * np.pi * G)

R_nu = 0.5

# set-up with nu cosmology
bg['rho_gamma'] = _rho_crit_100 * (1 - R_nu)
bg['rho_nu']    = _rho_crit_100 * R_nu


# set-up no nu cosmology
bg_no_nu = {}
bg_no_nu['rho_gamma'] = _rho_crit_100
bg_no_nu['rho_nu']    = 0

def rho_total(a, bg):
    #Total energy density at scale factor a [kg/m^3].
    return ((bg['rho_gamma'] + bg['rho_nu']) / a**4)

def hubble(a, bg):
    #Hubble parameter H(a) = sqrt(8 pi G / 3 * rho_total) in 1/Mpc."""
    return np.sqrt((8.0 * np.pi * G / 3.0) * rho_total(a, bg) * (Mpc_in_m / c_SI)**2 )



# ============================================================
# ADIABATIC INITIAL CONDITIONS
# ============================================================

# Curvature perturbation sets IC for all species
ZETA = -1/3

# truncate nu multipole hierarchy
LMAX = 100

# State vector
IDX_DG    = 0          # d_gamma
IDX_UG    = 1          # u_gamma
IDX_DN    = 2          # d_nu      = d_{0,nu}
IDX_UN    = 3          # u_nu      = d_{1,nu}
IDX_PIN   = 4          # pi_nu     = (2/3) d_{2,nu}
IDX_DLN_BASE = 5       # y[IDX_DLN_BASE + (l - 3)] = d_{l,nu} for l = 3..LMAX
NSTATE = IDX_DLN_BASE + (LMAX - 2)

def initial_state(R_nu):
    u = 5 * TAU0 * ZETA / (15 + 4 * R_nu)

    y = np.zeros(NSTATE)
    y[0] = -3 * ZETA
    y[1] = u
    y[2] = -3 * ZETA
    y[3] = u
    y[4] = TAU0**2 * 2 * ZETA / (3 * (15 + 4 * R_nu))

    return y



def metric_perturbations(d_g, u_g, d_n, u_n, pi_n, a, aH, bg):
    # metric perturbations we don't need to evolve
    # because they're determined algebraically from
    # species perturbations
    rho_g = bg['rho_gamma'] / a**4
    rho_n = bg['rho_nu']    / a**4

    # gamma_a = 4 pi G a^2 (rho_a + p_a)
    # For radiation, rho_a + p_a = (4/3)
    gamma_g = 4.0 * np.pi * G * a**2 * (4/3 * rho_g) * (Mpc_in_m / c_SI)**2
    gamma_n = 4.0 * np.pi * G * a**2 * (4/3 * rho_n) * (Mpc_in_m / c_SI)**2
    gamma_tot   = gamma_g + gamma_n


    # Total perturbations: x_a = (rho_a + p_a)/(rho + p) = gamma_a / gamma_tot.
    # So sum_a x_a y_a = (gamma_gamma * y_gamma + gamma_nu * y_nu) / gamma_tot.
    d_tot  = (gamma_g * d_g + gamma_n * d_n) / gamma_tot
    u_tot  = (gamma_g * u_g + gamma_n * u_n) / gamma_tot

    # Photons have pi_gamma = 0 in the perfect-fluid limit.
    pi_tot = (gamma_n * pi_n) / gamma_tot

    Psi = -gamma_tot * (d_tot + 3.0 * aH * u_tot) / (K**2 + 3.0 * gamma_tot)
    Phi = Psi - 3.0 * gamma_tot * pi_tot
    return Phi, Psi



# ============================================================
# EVOLUTION
#   Use scipy.solve_ivp to integrate linear perturbation eq.
# ============================================================

def rhs(tau, y, bg):
    a = hubble(1, bg) * tau
    aH = a * hubble(a, bg)

    # unpack state vector
    d_g, u_g, d_n, u_n, pi_n = y[:5]
    Phi, Psi = metric_perturbations(d_g, u_g, d_n, u_n, pi_n, a, aH, bg)

    dy = np.zeros(NSTATE)

    # photons
    dy[0] = -K**2 * u_g
    dy[1] = d_g / 3 + (Phi + Psi)
    
    # neutrino
    dy[2] = -K**2 * u_n
    dy[3] = d_n / 3 - K**2 * pi_n + (Phi + Psi)
    dy[4] = (4/15) * u_n - (2/5) * K**2 * y[5]
    for l in range(3, LMAX):
        d_lm1 = 1.5 * pi_n if l == 3 else y[l + 1]
        dy[l + 2] = (l / (2*l + 1)) * d_lm1 - ((l + 1) / (2*l + 1)) * K**2 * y[l + 3]
    dy[LMAX + 2] = y[LMAX + 1] - ((LMAX + 1) / tau) * y[LMAX + 2]

    return dy

# no nu
sol_no_nu = solve_ivp(rhs, (TAU0, TAU_END), initial_state(0), args=(bg_no_nu,),
                      method="LSODA", rtol=1e-10, atol=1e-12, dense_output=True)

# with nu
sol_wi_nu = solve_ivp(rhs, (TAU0, TAU_END), initial_state(R_nu), args=(bg,),
                      method="LSODA", rtol=1e-10, atol=1e-12, dense_output=True)


# Bashinsky, Seljak 2003 analytical result
dphi = 0.1912 * np.pi * R_nu
Dgamma = -0.2683 * R_nu
dg_bs = (1 + Dgamma) * sol_no_nu.sol(sol_no_nu.t + dphi * np.sqrt(3) / K)[0]
