# -*- coding: utf-8 -*-
"""
Chemostat simulation with Haldane-Contois kinetics
Simulation starting near a selected equilibrium
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =====================================================
# Estilo gráfico
# =====================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11
})

# =====================================================
# Parámetros
# =====================================================

mu_max = 1.0      # h^-1
Kc     = 0.1      # L/gX
Ki     = 20.0     # g/L

D      = 0.5      # 1/h
Sin    = 50.0     # g/L
Y      = 0.2      # gX/gS

# Tiempo
t_span = (0,25)
t_eval = np.linspace(*t_span,500)

# =====================================================
# Cinética híbrida Haldane-Contois
# =====================================================

def mu(S,X):
    return mu_max*S/(S + Kc*X + S**2/Ki)

# =====================================================
# Estados estacionarios
# =====================================================

equilibria = []

# ---------- Washout ----------

equilibria.append((0.0,Sin))

# ---------- Equilibrios con biomasa ----------

a = D/Ki
b = D*(1-Kc*Y) - mu_max
c = D*Kc*Y*Sin

disc = b**2 - 4*a*c

if disc >= 0:

    roots = sorted([
        (-b-np.sqrt(disc))/(2*a),
        (-b+np.sqrt(disc))/(2*a)
    ])

    for S_eq in roots:

        if -1e-8 <= S_eq <= Sin+1e-8:

            S_eq = np.clip(S_eq,0,Sin)
            X_eq = Y*(Sin-S_eq)

            equilibria.append((X_eq,S_eq))

print("\nEquilibria found:\n")

for i,(Xe,Se) in enumerate(equilibria):
    print(f"{i}: X*={Xe:.5f}, S*={Se:.5f}")

# =====================================================
# Condición inicial
# =====================================================

perturbation = 0.1

X_eq,S_eq = equilibria[2]

X0 = X_eq*(1+perturbation)
S0 = S_eq*(1+perturbation)

# ---------- washout ----------
# Equilibrio nulo
#
# X_eq,S_eq = equilibria[1]
# X0 = X_eq*(1+perturbation)
# S0 = S_eq*(1+perturbation)
#
# ---------- primer equilibrio con biomasa ----------
# Equilibrio estable
#
# X_eq,S_eq = equilibria[1]
# X0 = X_eq*(1+perturbation)
# S0 = S_eq*(1+perturbation)

# ---------- segundo equilibrio con biomasa ----------
# Equilibrio inestable
#
# X_eq,S_eq = equilibria[2]
# X0 = X_eq*(1+perturbation)
# S0 = S_eq*(1+perturbation)

# =====================================================
# Sistema dinámico
# =====================================================

def chemostat(t,y):

    X,S = y

    X = max(X,0)
    S = max(S,0)

    mu_val = mu(S,X)

    dXdt = (mu_val-D)*X
    dSdt = D*(Sin-S) - (1/Y)*mu_val*X

    return [dXdt,dSdt]

# =====================================================
# Simulación
# =====================================================

sol = solve_ivp(
    chemostat,
    t_span,
    [X0,S0],
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10
)

t = sol.t
X = sol.y[0]
S = sol.y[1]

mu_t = mu(np.maximum(S,0),np.maximum(X,0))

# =====================================================
# FIGURA 1
# Biomasa y sustrato
# =====================================================

fig, (ax1, ax3) = plt.subplots(
    2, 1,
    figsize=(6.5, 8.0),
    sharex=True
)

# -----------------------------------------------------
# Biomasa y sustrato
# -----------------------------------------------------

ax2 = ax1.twinx()

l1, = ax1.plot(
    t, X,
    color="black",
    linewidth=2.5,
    label="Biomass"
)

l2, = ax2.plot(
    t, S,
    color="gray",
    linewidth=2.3,
    label="Substrate"
)

ax1.set_ylabel(r"Biomass, $X$ (g/L)")
ax2.set_ylabel(r"Substrate, $S$ (g/L)")

margin = 0.05

ax1.set_xlim(t[0], t[-1])

X_min, X_max = np.min(X), np.max(X)
X_pad = max(margin * (X_max - X_min), 1e-3)
ax1.set_ylim(X_min - X_pad, X_max + X_pad + 1)

S_min, S_max = np.min(S), np.max(S)
S_pad = max(margin * (S_max - S_min), 1e-3)
ax2.set_ylim(S_min - S_pad, S_max + S_pad)

ax1.legend(
    [l1, l2],
    ["Biomass", "Substrate"],
    loc="upper center",
    ncol=2,
    frameon=False
)

# -----------------------------------------------------
# μ(t)
# -----------------------------------------------------

ax3.plot(
    t,
    mu_t,
    color="black",
    linewidth=2.5,
    label=r"$\mu(t)$"
)

ax3.axhline(
    D,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=r"$D$"
)

ax3.axhline(
    mu_max,
    color="gray",
    linestyle=":",
    linewidth=1.2,
    label=r"$\mu_{\max}$"
)

ax3.set_xlabel("Time (h)")
ax3.set_ylabel(r"Specific growth rate, $\mu$ (h$^{-1}$)")

mu_min_plot = min(np.min(mu_t), D, mu_max)
mu_max_plot = max(np.max(mu_t), D, mu_max)
mu_pad = max(margin * (mu_max_plot - mu_min_plot), 1e-3)

ax3.set_xlim(t[0], t[-1])
ax3.set_ylim(mu_min_plot - mu_pad,
             mu_max_plot + mu_pad)

ax3.legend(loc="upper right", frameon=False)

fig.suptitle(
    rf"Chemostat dynamics near equilibrium ($D={D:.2f}$ h$^{{-1}}$)"
)

fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    "Chemostat_Haldane_Contois_Dynamics.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "Chemostat_Haldane_Contois_Dynamics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# FIGURA 2
# Diagrama de fase
# =====================================================

fig, ax = plt.subplots(figsize=(6.5, 5.5))

# -----------------------------------------------------
# Campo vectorial
# -----------------------------------------------------

S_grid = np.linspace(0, Sin, 25)
X_grid = np.linspace(0, max(12, np.max(X) * 1.2), 25)

SS, XX = np.meshgrid(S_grid, X_grid)

MU = mu(SS, XX)

dX = (MU - D) * XX
dS = D * (Sin - SS) - (1 / Y) * MU * XX

# Normalizar flechas
N = np.sqrt(dX**2 + dS**2)
N[N == 0] = 1

ax.quiver(
    SS,
    XX,
    dS / N,
    dX / N,
    color="0.75",
    pivot="mid",
    scale=35
)

# -----------------------------------------------------
# Trayectoria simulada
# -----------------------------------------------------

ax.plot(
    S,
    X,
    color="black",
    linewidth=2.6,
    label="Trajectory"
)

# Flechas sobre la trayectoria
arrow_idx = np.linspace(40, len(S) - 20, 5, dtype=int)

for i in arrow_idx:

    ax.annotate(
        "",
        xy=(S[i + 8], X[i + 8]),
        xytext=(S[i], X[i]),
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1.7,
            mutation_scale=15
        )
    )

# -----------------------------------------------------
# Condición inicial
# -----------------------------------------------------

ax.plot(
    S0,
    X0,
    "ko",
    markersize=6,
    label="Initial condition"
)

# -----------------------------------------------------
# Puntos de equilibrio
# -----------------------------------------------------

for Xe, Se in equilibria:

    ax.plot(
        Se,
        Xe,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.7
    )

# -----------------------------------------------------
# Formato
# -----------------------------------------------------

ax.set_xlabel(r"Substrate concentration, $S$ (g/L)")
ax.set_ylabel(r"Biomass concentration, $X$ (g/L)")

ax.set_xlim(0, Sin)
ax.set_ylim(0, max(12, np.max(X) * 1.2))

ax.legend(frameon=False, loc="upper right")

plt.tight_layout()

plt.savefig(
    "Chemostat_Haldane_Contois_PhasePlane.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "Chemostat_Haldane_Contois_PhasePlane.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()