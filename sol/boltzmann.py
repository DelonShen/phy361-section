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

bg['n_H'] = (1 - cosmo['Y_He']) * bg['rho_b'] / m_H # number density of hydrogen today

 
def saha_xe(T, n_H):
    #Hydrogen Saha equilibrium x_e, `recombination.pdf` Eq. (1)
    #solving for xe with positive root of quadratic equation
    s = (CR * T)**1.5 * np.exp(-E_I / T) / n_H
    return (-s + np.sqrt(s * s + 4.0 * s)) / 2.0

def peebles_rhs(z, x_e, bg, return_C = False):
    #Peebles ODE RHS of dx_e / dz, `recombination.pdf` Eq. (3)

    a = 1.0 / (1.0 + z)
    T = cosmo['T_cmb'] * (1.0 + z)
    n_H = bg['n_H'] * (1.0 + z)**3

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
nz = 20000
z_arr = np.linspace(6000, 0, nz)
xe_arr = np.empty(nz)

# find redshift where x_e first drops below 0.99
# to set IC for Peebles ODE
IC_idx = None

for i, z in enumerate(z_arr):
    T = cosmo['T_cmb'] * (1.0 + z)
    n_H = bg['n_H'] * (1.0 + z)**3
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


# ============================================================
# REIONIZATION
#   tanh model for H and He reionization mimicing CAMB
# ============================================================


bg['f_He'] = cosmo['Y_He'] / ((m_He4 / m_H) * (1.0 - cosmo['Y_He'])) # n(He) / n(H)
akthom = sigma_T * bg['n_H'] * Mpc_in_m   # (n_P sigma_T) today, [Mpc^-1]

# Reionization parameters (following CAMB defaults)
_REION_DELTA_Z = 0.5         # H reion tanh width
_REION_ZEXP = 1.5            # tanh argument exponent: tanh in (1+z)^{3/2}
_HE_REION_Z = 3.5            # He+ -> He++ midpoint
_HE_REION_DZ = 0.4           # He+ -> He++ width



def _tanh_step_in_y(z, z_mid, delta_z, amplitude):
    #tanh stepy used as H + He reionization model
    y = (1.0 + z)**_REION_ZEXP
    y_mid = (1.0 + z_mid)**_REION_ZEXP
    delta_y = (_REION_ZEXP * (1.0 + z_mid)**(_REION_ZEXP - 1.0) * delta_z)
    arg = (y_mid - y) / delta_y
    return 0.5 * amplitude * (1.0 + np.tanh(arg))


def reion_xe(z, z_re, f_He):
    # tanh model for reionization of 
    # (a) HI->HII + HeI->HeII and (b) HeII->HeIII
    return (_tanh_step_in_y(z, z_re, _REION_DELTA_Z, 1.0 + f_He)    # (a)
            + _tanh_step_in_y(z, _HE_REION_Z, _HE_REION_DZ, f_He))  # (b)


def thomson_opacity(z_arr, x_e_arr, bg):
    #Thomson opacity in conformal time:
    a_arr = 1.0 / (1.0 + np.asarray(z_arr, dtype=float))
    return x_e_arr * akthom / a_arr**2


def optical_depth(eta_arr, tau_dot_arr):
    # Optical depth tau(eta), assumes eta_arr is sorted s.t. z(eta) is decreasing
    # optical_depth[0] returns tau at smallest eta (largest z) = total
    return -scipy.integrate.cumulative_trapezoid(tau_dot_arr[::-1], eta_arr[::-1], initial=0.0,)[::-1]


def find_z_re(target_tau, bg):
    #find z_re such that optical depth to reionization matches input cosmology with binary search

    f_He = bg['f_He']
    
    # grid for computing tau integral
    z_max = 30.0 + 8.0 * _REION_DELTA_Z
    z_grid = np.linspace(z_max, 0.0, 2000)
    a_grid = 1.0 / (1.0 + z_grid)
    eta_grid = conformal_time(a_grid, bg)


    def tau_reion(z_re):
        x_e = reion_xe(z_grid, z_re, f_He)
        tau_dot = thomson_opacity(z_grid, x_e, bg)
        # reion optical depth integrated back to the present.
        return optical_depth(eta_grid, tau_dot)[0]

    # define bounds of binary search
    z_lo, z_hi = 2.0, 30.0
    tau_bot, tau_top = None, None
    for _ in range(1000):
        z_mid = 0.5 * (z_lo + z_hi)
        tau_mid = tau_reion(z_mid)
        if tau_mid > target_tau:
            z_hi, tau_top = z_mid, tau_mid
        else:
            z_lo, tau_bot = z_mid, tau_mid
        if abs(z_hi - z_lo) < 1e-3:
            break

    return z_hi


# compute reionization history
z_re = find_z_re(cosmo['tau_reion'], bg)
x_e_reion = reion_xe(z_arr, z_re, bg['f_He'])

# include reionization in ionization history 
xe_arr = np.maximum(x_e_reion, xe_arr)

# ============================================================
# PERTURBATIONS
# ============================================================

# interpolating optical depth and thompson opacity
_a_pert = np.logspace(-10, 0, 4000)
_tau_pert = conformal_time(_a_pert, bg)
_z_pert = 1.0 / _a_pert - 1.0

_xe_pert = np.where(
    _z_pert > z_arr[0],
    1.0, # assume fully ionized for z > z_arr[0]
    np.interp(_z_pert, z_arr[::-1], xe_arr[::-1]),
)
_tau_dot_pert = thomson_opacity(_z_pert, _xe_pert, bg)   # 1/Mpc

def scale_factor_from_tau(tau):
    return np.interp(tau, _tau_pert, _a_pert)

def tau_dot_of_tau(tau):
    return np.interp(tau, _tau_pert, _tau_dot_pert)


# state vector for TCA
LMAX_G = 15       # photon multipoles
LMAX_N = 15       # massless-neutrino multipoles

IDX_DC,  IDX_UC  = 0, 1                 # CDM:    d_c, u_c
IDX_DB,  IDX_UB  = 2, 3                 # baryon: d_b, u_b 
IDX_DG,  IDX_UG, IDX_PIG = 4, 5, 6      # photon monopole/dipole/quadrupole
IDX_DLG_BASE = 7                        # d_{l,g} for l = 3..LMAX_G
_NG = LMAX_G - 2

IDX_DN  = IDX_DLG_BASE + _NG            # neutrino d_n
IDX_UN  = IDX_DN + 1                    # neutrino u_n
IDX_PIN = IDX_DN + 2                    # neutrino pi_n
IDX_DLN_BASE = IDX_PIN + 1              # d_{l,n} for l = 3..LMAX_N
_NN = LMAX_N - 2

NSTATE = IDX_DLN_BASE + _NN


ZETA = -1.0 # chosen to match default CAMB convention, really we're just computing transfer function here


def metric_perturbations(d_g, u_g, pi_g, d_n, u_n, pi_n,
                         d_c, u_c, d_b, u_b, a, aH, k, bg):
    rho_g = bg['rho_gamma'] / a**4
    rho_n = bg['rho_nu']    / a**4
    rho_c = bg['rho_c']     / a**3
    rho_b = bg['rho_b']     / a**3

    factor  = 4.0 * np.pi * G * a**2 * (Mpc_in_m / c_SI)**2

    gamma_g = factor * (4.0/3.0 * rho_g)
    gamma_n = factor * (4.0/3.0 * rho_n)
    gamma_c = factor * rho_c
    gamma_b = factor * rho_b
    gamma_tot = gamma_g + gamma_n + gamma_c + gamma_b

    d_tot  = (gamma_g*d_g + gamma_n*d_n + gamma_c*d_c + gamma_b*d_b) / gamma_tot
    u_tot  = (gamma_g*u_g + gamma_n*u_n + gamma_c*u_c + gamma_b*u_b) / gamma_tot
    pi_tot = (gamma_g * pi_g + gamma_n * pi_n) / gamma_tot

    Psi = -gamma_tot * (d_tot + 3.0 * aH * u_tot) / (k**2 + 3.0 * gamma_tot)
    Phi = Psi - 3.0 * gamma_tot * pi_tot
    return Phi, Psi


def initial_state(tau_init, k, bg):
    R_nu = bg['rho_nu'] / (bg['rho_gamma'] + bg['rho_nu'])
    u    = 5.0 * tau_init * ZETA / (15.0 + 4.0 * R_nu)

    y = np.zeros(NSTATE)
    y[IDX_DC]  = -3.0 * ZETA
    y[IDX_UC]  = u

    y[IDX_DB]  = -3.0 * ZETA
    y[IDX_UB]  = u                          # u_b = u_g at IC (TCA)

    y[IDX_DG]  = -3.0 * ZETA
    y[IDX_UG]  = u
    # y[IDX_PIG] = 0; higher d_{l,g} = 0; higher d_{l,n} = 0 (set by np.zeros)

    y[IDX_DN]  = -3.0 * ZETA
    y[IDX_UN]  = u
    y[IDX_PIN] = tau_init**2 * 2.0 * ZETA / (3.0 * (15.0 + 4.0 * R_nu))

    return y



def _cdm_rhs(y, k, aH, Phi, dy):
    # same in TCA and full system
    # so we make it its own function
    dy[IDX_DC] = -k**2 * y[IDX_UC]
    dy[IDX_UC] = -aH * y[IDX_UC] + Phi
    return dy


def _neutrino_rhs(y, k, tau, Phi, Psi, dy):
    # same in TCA and full system
    # so we make it its own function
    d_n  = y[IDX_DN]
    u_n  = y[IDX_UN]
    pi_n = y[IDX_PIN]

    dy[IDX_DN]  = -k**2 * u_n
    dy[IDX_UN]  = d_n / 3.0 - k**2 * pi_n + (Phi + Psi)
    dy[IDX_PIN] = (4.0/15.0) * u_n - (2.0/5.0) * k**2 * y[IDX_DLN_BASE]

    # higher-l recursion (l = 3..LMAX_N - 1)
    for l in range(3, LMAX_N):
        d_lm1 = 1.5 * pi_n if l == 3 else y[IDX_DLN_BASE + (l - 4)]
        d_lp1 = y[IDX_DLN_BASE + (l - 2)]
        dy[IDX_DLN_BASE + (l - 3)] = (l/(2*l+1)) * d_lm1 - ((l+1)/(2*l+1)) * k**2 * d_lp1

    dy[IDX_DLN_BASE + (LMAX_N - 3)] = (
        y[IDX_DLN_BASE + (LMAX_N - 4)]
        - ((LMAX_N + 1) / tau) * y[IDX_DLN_BASE + (LMAX_N - 3)]
    )

    return dy


def tca_rhs(tau, y, k, bg):
    # very similar to what we did last week in `nu_phase_shift.py`
    a  = scale_factor_from_tau(tau)
    aH = a * hubble(a, bg)

    d_g = y[IDX_DG]; u_g = y[IDX_UG]
    d_b = y[IDX_DB]
    d_c = y[IDX_DC]; u_c = y[IDX_UC]
    d_n = y[IDX_DN]; u_n = y[IDX_UN]; pi_n = y[IDX_PIN]
    u_b = u_g

    Phi, Psi = metric_perturbations(d_g, u_g, 0.0, d_n, u_n, pi_n,
                                     d_c, u_c, d_b, u_b, a, aH, k, bg)

    rho_g = bg['rho_gamma'] / a**4
    rho_b = bg['rho_b']     / a**3
    R_b   = 3.0 * rho_b / (4.0 * rho_g)

    tau_dot = tau_dot_of_tau(tau)
    tau_c = 1.0 / tau_dot
    tau_d = (tau_c / 6.0) * (1.0 - 14.0/(15.0*(1.0+R_b)) + 1.0/(1.0+R_b)**2)
    friction = aH * R_b / (1.0 + R_b) + 2.0 * k**2 * tau_d

    dy = np.zeros(NSTATE)

    # Photon: continuity + TCA
    dy[IDX_DG] = -k**2 * u_g
    dy[IDX_UG] = (-friction * u_g
                  + d_g / (3.0 * (1.0 + R_b))
                  + (Phi + Psi / (1.0 + R_b)))

    # Baryon: locked to photons
    dy[IDX_DB] = -k**2 * u_g
    dy[IDX_UB] = dy[IDX_UG]

    dy = _cdm_rhs(y, k, aH, Phi, dy)
    dy = _neutrino_rhs(y, k, tau, Phi, Psi, dy)
    return dy


def solve_perturbations_tca(k, tau_max=bg['tau0'],):
    tau_init = 1e-3 / k

    y0 = initial_state(tau_init, k, bg)
    sol_tca = scipy.integrate.solve_ivp(
        tca_rhs, (tau_init, tau_max), y0,
        args=(k, bg), method='LSODA',
        rtol=1e-8, atol=1e-10, dense_output=True,
    )

    return sol_tca

# The above is enough to test code against CAMB in the tight-coupling regime
# I do this in `section06_tests_TCA.py` 

def full_rhs(tau, y, k, bg):
    a  = scale_factor_from_tau(tau)
    aH = a * hubble(a, bg)

    d_c = y[IDX_DC]; u_c = y[IDX_UC]
    d_b = y[IDX_DB]; u_b = y[IDX_UB]
    d_g = y[IDX_DG]; u_g = y[IDX_UG]; pi_g = y[IDX_PIG]
    d_n = y[IDX_DN]; u_n = y[IDX_UN]; pi_n = y[IDX_PIN]

    Phi, Psi = metric_perturbations(d_g, u_g, pi_g, d_n, u_n, pi_n,
                                     d_c, u_c, d_b, u_b, a, aH, k, bg)

    rho_g = bg['rho_gamma'] / a**4
    rho_b = bg['rho_b']     / a**3
    R_b   = 3.0 * rho_b / (4.0 * rho_g)


    tau_dot = tau_dot_of_tau(tau)

    dy = np.empty(NSTATE)

    dy = _cdm_rhs(y, k, aH, Phi, dy)

    # Baryon: continuity + Euler
    dy[IDX_DB] = -k**2 * u_b
    dy[IDX_UB] = -aH * u_b + Phi + 1 / R_b * tau_dot * (u_g - u_b)

    # Photon: continuity + Euler
    dy[IDX_DG] = -k**2 * u_g
    dy[IDX_UG] = (d_g / 3.0 - k**2 * pi_g + (Phi + Psi)
                  + tau_dot * (u_b - u_g))

    # Photon quadrupole
    dy[IDX_PIG] = (4.0/15.0) * u_g - (2.0/5.0) * k**2 * y[IDX_DLG_BASE] - tau_dot * pi_g

    # higher-l photon recursion (l = 3..LMAX_G - 1)
    for l in range(3, LMAX_G):
        d_lm1 = 1.5 * pi_g if l == 3 else y[IDX_DLG_BASE + (l - 4)]
        d_lp1 = y[IDX_DLG_BASE + (l - 2)]
        dy[IDX_DLG_BASE + (l - 3)] = (
            (l/(2*l+1)) * d_lm1 - ((l+1)/(2*l+1)) * k**2 * d_lp1
            - tau_dot * y[IDX_DLG_BASE + (l - 3)]
        )

    dy[IDX_DLG_BASE + (LMAX_G - 3)] = (
        y[IDX_DLG_BASE + (LMAX_G - 4)]
        - ((LMAX_G + 1) / tau) * y[IDX_DLG_BASE + (LMAX_G - 3)]
        - tau_dot * y[IDX_DLG_BASE + (LMAX_G - 3)]
    )

    dy = _neutrino_rhs(y, k, tau, Phi, Psi, dy)
    return dy



def solve_perturbations(k, tau_max=bg['tau0']):
    tau_init = 1e-3 / k

    # Find when tau_c * k > 0.02.
    lo, hi = tau_init, tau_max
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        if k / tau_dot_of_tau(mid) > 0.02:
            hi = mid
        else:
            lo = mid
    tau_switch = hi

    # Stage 1: TCA
    y0 = initial_state(tau_init, k, bg)
    sol_tca = scipy.integrate.solve_ivp(
        tca_rhs, (tau_init, tau_switch), y0,
        args=(k, bg), method='LSODA',
        rtol=1e-8, atol=1e-10, dense_output=True,
    )

    # Stage 2: full
    sol_full = scipy.integrate.solve_ivp(
        full_rhs, (tau_switch, tau_max), sol_tca.y[:, -1],
        args=(k, bg), method='LSODA',
        rtol=1e-8, atol=1e-10, dense_output=True,
    )

    return sol_tca, sol_full
