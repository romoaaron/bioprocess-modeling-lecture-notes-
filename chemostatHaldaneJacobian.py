# -*- coding: utf-8 -*-
"""
Eigenvalue analysis of the Haldane–Contois chemostat
Correct Jacobian
"""

import numpy as np

# =====================================================
# Parámetros
# =====================================================

mu_max = 1.0      # h^-1
Kc = 0.1          # L/gX
Ki = 20.0         # g/L

D = 0.5           # h^-1
Sin = 50.0        # g/L
Y = 0.2           # gX/gS

# =====================================================
# Cinética Haldane–Contois
# =====================================================

def mu(S, X):
    return mu_max * S / (S + Kc * X + S**2 / Ki)

# =====================================================
# Derivadas parciales
# =====================================================

def mu_X(S, X):
    denom = S + Kc * X + S**2 / Ki
    return -mu_max * Kc * S / denom**2

def mu_S(S, X):
    denom = S + Kc * X + S**2 / Ki
    return mu_max * (Kc * X - S**2 / Ki) / denom**2

# =====================================================
# Estados estacionarios
# =====================================================

equilibria = [(0.0, Sin)]      # Washout

a = D / Ki
b = D * (1 - Kc * Y) - mu_max
c = D * Kc * Y * Sin

disc = b**2 - 4 * a * c

if disc >= 0:

    roots = sorted([
        (-b - np.sqrt(disc)) / (2 * a),
        (-b + np.sqrt(disc)) / (2 * a)
    ])

    for S_eq in roots:

        if -1e-8 <= S_eq <= Sin + 1e-8:

            S_eq = np.clip(S_eq, 0, Sin)
            X_eq = Y * (Sin - S_eq)

            equilibria.append((X_eq, S_eq))

# =====================================================
# Jacobiano CORREGIDO
# =====================================================

def jacobian(X, S):

    mu_val = mu(S, X)
    mux = mu_X(S, X)
    mus = mu_S(S, X)

    J = np.array([
        [(mu_val - D) + X * mux,
         X * mus],

        [-(mu_val + X * mux) / Y,
         -D - (X * mus) / Y]
    ])

    return J

# =====================================================
# Análisis
# =====================================================

print("\nEigenvalue analysis\n")

for i, (Xeq, Seq) in enumerate(equilibria):

    J = jacobian(Xeq, Seq)
    eig = np.linalg.eigvals(J)

    print("=" * 45)
    print(f"Equilibrium E{i}")
    print(f"X* = {Xeq:.6f}")
    print(f"S* = {Seq:.6f}")

    print("\nJacobian:")
    print(np.round(J, 6))

    print("\nEigenvalues:")
    print(np.round(eig, 6))

    if np.all(np.real(eig) < 0):
        print("Classification: Locally stable.")
    elif np.any(np.real(eig) > 0) and np.any(np.real(eig) < 0):
        print("Classification: Saddle point.")
    elif np.all(np.real(eig) > 0):
        print("Classification: Unstable.")
    else:
        print("Classification: Marginal / requires further analysis.")