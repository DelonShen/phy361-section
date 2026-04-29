import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family" : "serif",
    'figure.figsize': (5, 6),
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

from pancake import *



target_a    = [0.1, 0.5, 1.0]
A_zel       = 1.0 / (a_cross * 2.0 * np.pi)

snap_a = np.array([s['a'] for s in snapshots])



fig, axes = plt.subplots(
    2, 3, figsize=(6, 6/1.618), sharex=True, sharey='row',
)

for col, (a_t) in enumerate(target_a):
    ax_d = axes[0, col]
    ax_p = axes[1, col]

    idx  = int(np.argmin(np.abs(snap_a - a_t)))
    snap = snapshots[idx]
    a    = snap['a']
    xN   = snap['x']
    pN   = snap['p']

    # Zeldovich:  x_zel(q,a) = q + a A sin(2 pi q)
    #             p_zel(q,a) = a^{3/2} A sin(2 pi q)
    x_zel = zeldovich_x(q, a, A_zel)
    p_zel = zeldovich_p(q, a, A_zel)

    rho_N   = CIC_deposit(xN,                  N_grid)
    rho_zel = CIC_deposit(np.mod(x_zel, 1.0),  N_grid)
    x_grid  = (np.arange(N_grid) + 0.5) / N_grid

    # density contrast normalized by scale factor: delta/a
    delta_over_a_N   = (rho_N   - 1.0) / a
    delta_over_a_zel = (rho_zel - 1.0) / a

    # ---- top: density contrast / a
    ax_d.plot(x_grid, delta_over_a_N,   color='k', ls='-',  lw=1.0)
    ax_d.plot(x_grid, delta_over_a_zel, color='r', ls='-', lw=1.0/1.618)
    ax_d.set_title(r'$a=%.1f$' % a, fontsize=14)

    # ---- bottom: phase space
    ax_p.scatter(xN[::len(xN)//100], pN[::len(xN)//100], color='k', s=1, marker='.')
    ax_p.plot(x_zel, p_zel, color='r', ls='-', lw=1.0/1.618)

axes[0, 0].set_ylabel(r'$\delta(\tilde x)/a$')
axes[1, 0].set_ylabel(r'$\tilde p$')
for col in range(len(target_a)):
    axes[1, col].set_xlabel(r'$\tilde x$')
    axes[1, col].set_xlim(0, 1)
    axes[1, col].set_xticks([])
    axes[1, col].set_ylim(-0.38, 0.38)
    axes[1, col].set_yticks([-0.3, 0, 0.3])

    axes[0, col].set_ylim(-1, 10)

# linestyle legend in the leftmost density panel
style_handles = [
    plt.Line2D([0], [0], color='k', ls='-',  lw=1.0, label='N-body'),
    plt.Line2D([0], [0], color='r', ls='-', lw=1.0, label='Zeldovich'),
]
axes[1, 1].legend(
    handles=style_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    frameon=False,
    fontsize=14,
    ncol=2,
)

plt.savefig('figures/section04_pancake.pdf', bbox_inches='tight')
plt.close()



# ============================================================
# MOVIE: ALL SNAPSHOTS
# ============================================================

from matplotlib.animation import FuncAnimation

fig_m, (ax_dm, ax_pm) = plt.subplots(2, 1, figsize=(6, 6/1.618), sharex=True)

x_grid = (np.arange(N_grid) + 0.5) / N_grid

# fixed axes so the frame doesn't jump
ax_dm.set_xlim(0, 1)
ax_dm.set_ylabel(r'$\delta(\tilde x)/a$')
ax_pm.set_xlabel(r'$\tilde x$')
ax_pm.set_ylabel(r'$\tilde p$')

p_max = max(np.max(np.abs(s['p'])) for s in snapshots)
ax_pm.set_ylim(-1.1 * p_max, 1.1 * p_max)

# pre-compute the y-range for delta/a from a representative late snapshot
ax_dm.set_ylim(-1, 20)

line_d_N,   = ax_dm.plot([], [], color='k', ls='-',  lw=1.0,         label='N-body')
line_d_zel, = ax_dm.plot([], [], color='r', ls='-', lw=1.0/1.618,   label='Zeldovich')
scat_p_N    = ax_pm.scatter([], [], color='k', s=1, marker='.')
line_p_zel, = ax_pm.plot([], [], color='r', ls='-', lw=1.0/1.618)
title       = ax_dm.set_title('')
ax_dm.legend(loc='upper left', frameon=False, fontsize=9)

def update(frame):
    snap = snapshots[frame]
    a    = snap['a']
    xN   = snap['x']
    pN   = snap['p']

    x_zel_f = zeldovich_x(q, a, A_zel)
    p_zel_f = zeldovich_p(q, a, A_zel)

    rho_N   = CIC_deposit(xN,                    N_grid)
    rho_zel = CIC_deposit(np.mod(x_zel_f, 1.0),  N_grid)

    line_d_N.set_data(  x_grid, (rho_N   - 1.0) / a)
    line_d_zel.set_data(x_grid, (rho_zel - 1.0) / a)

    scat_p_N.set_offsets(np.column_stack([xN[::len(xN)//100], pN[::len(xN)//100]]))
    line_p_zel.set_data(x_zel_f, p_zel_f)

    title.set_text(r'$a=%.3f$' % a)
    return line_d_N, line_d_zel, scat_p_N, line_p_zel, title


anim = FuncAnimation(fig_m, update, frames=len(snapshots), blit=False)
anim.save('figures/section04_pancake.mp4', fps=120, dpi=300)

plt.close(fig_m)



# ============================================================
# TWO-STREAM PLASMA
# ============================================================

N_full = len(plasma_snapshots[0]['x'])
N_half = N_full // 2
order  = np.random.default_rng(0).permutation(N_full)
colors = np.where(order < N_half, 'b', 'r')




# ---- snapshot panel ------------------------------------------

target_t = [1.0, 25.0, 50.0]
snap_t   = np.array([s['t'] for s in plasma_snapshots])

fig, axes = plt.subplots(1, 3, figsize=(6, 6/1.618/1.618),
                         sharex=True, sharey=True)

for col, t_t in enumerate(target_t):
    ax   = axes[col]
    idx  = int(np.argmin(np.abs(snap_t - t_t)))
    snap = plasma_snapshots[idx]

    ax.scatter(snap['x'][order], snap['v'][order], marker='.',
               s=1.0, c=colors, lw = 0, rasterized=True)
    ax.set_title(r'$t=%.1f$' % snap['t'], fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(-.19, .19)
    ax.set_xlabel(r'$\tilde x$')
    ax.set_xticks([])

axes[0].set_ylabel(r'$\tilde v$')

style_handles = [
    plt.Line2D([0], [0], color='b', ls='-', label=r'$+v_s$ stream'),
    plt.Line2D([0], [0], color='r',  ls='-', label=r'$-v_s$ stream'),
]
axes[1].legend(
    handles=style_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    frameon=False,
    fontsize=14,
    ncol = 2
)

plt.savefig('figures/section04_two_stream.pdf', bbox_inches='tight')
plt.close()



# ---- movie ---------------------------------------------------

fig_m, ax_m = plt.subplots(figsize=(2, 6/1.618/1.618))

ax_m.set_xlim(0, 1)
ax_m.set_ylim(-.19, .19)
ax_m.set_xlabel(r'$\tilde x$')
ax_m.set_xticks([])
ax_m.set_yticks([])
ax_m.set_ylabel(r'$\tilde v$')

s0      = plasma_snapshots[0]
scat_p  = ax_m.scatter(s0['x'][order], s0['v'][order],
                       s=0.5, c=colors, lw = 0, rasterized=True, marker='.')
title_p = ax_m.set_title('')

def update_plasma(frame):
    snap = plasma_snapshots[frame]
    scat_p.set_offsets(np.column_stack([snap['x'][order], snap['v'][order]]))
    title_p.set_text(r'$t=%.2f$' % snap['t'])
    return scat_p, title_p


anim_p = FuncAnimation(fig_m, update_plasma, frames=len(plasma_snapshots), blit=False)
anim_p.save('figures/section04_two_stream.mp4', fps=300, dpi = 300)

plt.close(fig_m)
