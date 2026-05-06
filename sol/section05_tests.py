from nu_phase_shift import *
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


plt.plot(sol_no_nu.t, sol_no_nu.y[0], 'lightgrey', label=r'$d_{\gamma}\, ({\rm no}\,\nu)$')
plt.plot(sol_wi_nu.t, sol_wi_nu.y[2], 'b', label=r'$d_{\nu}$')
plt.plot(sol_wi_nu.t, sol_wi_nu.y[0], 'k', label=r'$d_{\gamma}$')
plt.plot(sol_no_nu.t, dg_bs, 'r--', label=r'Analytical $d_{\gamma}$')



plt.xlabel(r'$\tau\,[{\rm Mpc}]$')
plt.xlim(TAU0, 8)
plt.ylabel(r'$d_a$')
plt.ylim(-1.5, 1.5,)


plt.legend(frameon=False, ncols=4, bbox_to_anchor=(0.5, 1.0), loc='lower center')

plt.savefig('figures/section05_nu_phase_shift.pdf', bbox_inches='tight')
