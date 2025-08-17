import numpy as np
from numba import jit, int64, float64
from constants import (
    NU, NU_DEP, E_B_W, E_B_RE, E_B_C, E_DIFF_W, E_DIFF_RE, E_DIFF_C,
    K_T, T_MELT, RATE_THRESHOLD, ANISOTROPY_FACTOR, IMPURITY_RE,
    CARBON_SOLUTION_ENERGY, I0, DELTA_T_C, K_NUC, BETA_IMP_NUC, MAX_IMP_FRACTION
)

@jit(nopython=True, cache=True)
def compute_misorientation(theta1, phi1, theta2, phi2):
    v1 = np.array([
        np.sin(theta1) * np.cos(phi1),
        np.sin(theta1) * np.sin(phi1),
        np.cos(theta1)
    ])
    v2 = np.array([
        np.sin(theta2) * np.cos(phi2),
        np.sin(theta2) * np.sin(phi2),
        np.cos(theta2)
    ])
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    dot = max(min(dot, 1.0), -1.0)
    return np.arccos(dot)

@jit(nopython=True, cache=True)
def get_bcc_neighbors(i, j, k, L):
    neighbors = np.zeros((14, 3), dtype=int64)
    count = 0
    for di, dj, dk in [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                       (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
            neighbors[count] = [ni, nj, nk]
            count += 1
    for di, dj, dk in [(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2)]:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
            neighbors[count] = [ni, nj, nk]
            count += 1
    return neighbors[:count]

@jit(nopython=True, cache=True)
def compute_row_events(i, state, orientation_theta, orientation_phi, T,
                      atom_type, defects, nu, nu_dep, E_b, E_diff, kT, T_melt,
                      I0, delta_T_c, top_layer, L, states_w, states_re, states_c,
                      impurity_c):
    events = []
    row_state = state[i]
    row_theta = orientation_theta[i]
    row_phi = orientation_phi[i]
    row_T = T[i]
    row_defects = defects[i]

    # Deposition
    if i == top_layer:
        y_indices, z_indices = np.where(row_state == 0)
        for idx in range(len(y_indices)):
            j = y_indices[idx]
            k = z_indices[idx]
            local_T = max(row_T[j,k], 1.0)  # Prevent division by zero
            thermal_factor = np.exp(-(T_melt - local_T) / (kT * local_T))
            effective_nu_dep = nu_dep * thermal_factor
            if not np.isfinite(effective_nu_dep):
                continue
            r = np.random.random()
            if r < impurity_c:
                atom = states_c
            elif r < impurity_c + IMPURITY_RE:
                atom = states_re
            else:
                atom = states_w
            events.append((b'dep', (i,j,k), effective_nu_dep, (-1,-1,-1), atom))

    # Occupied site events
    occ_indices = np.where(row_state != 0)
    for idx in range(len(occ_indices[0])):
        j = occ_indices[0][idx]
        k = occ_indices[1][idx]
        atom = row_state[j,k]
        if atom == 4:  # Defect state
            continue

        if atom == states_w:
            E_b_atom = E_b[0]
            E_diff_atom = E_diff[0]
        elif atom == states_re:
            E_b_atom = E_b[1]
            E_diff_atom = E_diff[1]
        else:  # states_c
            E_b_atom = E_b[2]
            E_diff_atom = E_diff[2]

        local_T = max(row_T[j,k], 1.0)
        defect_factor = 1.0 + row_defects[j,k]
        neighbors = get_bcc_neighbors(i, j, k, L)
        n_bonds = 0
        for ni, nj, nk in neighbors:
            if state[ni, nj, nk] != 0:
                n_bonds += 1
        for ni, nj, nk in neighbors:
            if state[ni, nj, nk] == 0:
                neighbor_T = max(T[ni, nj, nk], 1.0)
                dT = abs(local_T - neighbor_T)
                denom = max(T_melt - neighbor_T, 1.0)
                grad_factor = 1.0 + 0.1 * dT / denom
                E_diff_total = max(E_diff_atom + 0.1 * n_bonds * E_b_atom, 0.0)  # Prevent negative
                rate_diff = nu * grad_factor * np.exp(-defect_factor * E_diff_total / (kT * local_T))
                if rate_diff > RATE_THRESHOLD and np.isfinite(rate_diff):
                    events.append((b'diff', (i,j,k), rate_diff, (ni,nj,nk), atom))

    # Empty site events (nucleation + attachment)
    y_indices, z_indices = np.where(row_state == 0)
    for idx in range(len(y_indices)):
        j = y_indices[idx]
        k = z_indices[idx]
        local_T = max(row_T[j,k], 1.0)
        dT = T_melt - local_T

        # Nucleation
        if dT > delta_T_c:
            neighbors = get_bcc_neighbors(i, j, k, L)
            n_imp = 0
            for ni, nj, nk in neighbors:
                if state[ni, nj, nk] in (states_re, states_c):
                    n_imp += 1
            f_imp = min(MAX_IMP_FRACTION, n_imp / max(len(neighbors), 1))
            K_eff = K_NUC * (1.0 - BETA_IMP_NUC * f_imp)
            K_eff = max(0.1 * K_NUC, min(K_NUC, K_eff))
            barrier = K_eff / max((dT + 1e-6) * (dT + 1e-6), 1e-6)  # Prevent div by zero
            rate_nuc = I0 * np.exp(-barrier / (kT * local_T))
            if rate_nuc > RATE_THRESHOLD and np.isfinite(rate_nuc):
                events.append((b'nuc', (i,j,k), rate_nuc, (-1,-1,-1), states_w))

        # Attachment
        neighbors = get_bcc_neighbors(i, j, k, L)
        for ni, nj, nk in neighbors:
            if state[ni, nj, nk] != 0:
                neighbor_atom = state[ni, nj, nk]
                if neighbor_atom == states_w:
                    ia_n = 0
                elif neighbor_atom == states_re:
                    ia_n = 1
                elif neighbor_atom == states_c:
                    ia_n = 2
                else:
                    continue
                mis_ang = compute_misorientation(
                    row_theta[j,k], row_phi[j,k],
                    orientation_theta[ni,nj,nk], orientation_phi[ni,nj,nk]
                )
                k_minus = max(k - 1, 0)
                k_plus = min(k + 1, L - 1)
                grad_z = (T[i,j,k_plus] - T[i,j,k_minus]) * 0.5
                grad_factor = max(0.0, grad_z) / max(T_melt - local_T, 1.0)
                E_att = 0.5 * E_b[ia_n] * (1.0 - np.cos(mis_ang))
                rate_att = nu * np.exp(-E_att / (kT * local_T)) * (1.0 + ANISOTROPY_FACTOR * grad_factor)
                if rate_att > RATE_THRESHOLD and np.isfinite(rate_att):
                    events.append((b'att', (i,j,k), rate_att, (ni,nj,nk), neighbor_atom))

    return events

def get_event_rates(state, orientation_theta, orientation_phi, T, atom_type, defects_mask, L,
                    states_w, states_re, states_c, step=0, debug_step=1000, impurity_c=0.0):
    E_b = np.array([E_B_W, E_B_RE, E_B_C])
    E_diff = np.array([E_DIFF_W, E_DIFF_RE, E_DIFF_C])
    top_layer = L - 1
    events = []
    for i in range(L):
        row_events = compute_row_events(
            i, state, orientation_theta, orientation_phi, T,
            atom_type, defects_mask, NU, NU_DEP, E_b, E_diff, K_T, T_MELT,
            I0, DELTA_T_C, top_layer, L, states_w, states_re, states_c, impurity_c
        )
        for event in row_events:
            events.append((event[0], event[1], event[2], event[3], event[4]))
    return events