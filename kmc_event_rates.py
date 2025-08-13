import numpy as np
from numba import jit
from constants import (
    LATTICE_SIZE, NU, NU_DEP,
    E_B_W, E_B_RE, E_B_C,
    E_DIFF_W, E_DIFF_RE, E_DIFF_C,
    K_T, T_MELT,
    RATE_THRESHOLD, ANISOTROPY_FACTOR, EPSILON, IMPURITY_RE,
    CARBON_SOLUTION_ENERGY
)

@jit(nopython=True, cache=True)
def get_bcc_neighbors(i, j, k, L):
    neighbors = []
    # 1st nearest neighbors in BCC
    for di, dj, dk in [
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    ]:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
            neighbors.append((ni, nj, nk))
    # 2nd nearest neighbors in BCC
    for di, dj, dk in [
        (2, 0, 0), (-2, 0, 0),
        (0, 2, 0), (0, -2, 0),
        (0, 0, 2), (0, 0, -2)
    ]:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
            neighbors.append((ni, nj, nk))
    return neighbors

@jit(nopython=True, cache=True)
def compute_misorientation(theta1, phi1, theta2, phi2):
    dot = (
        np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)
        + np.cos(theta1) * np.cos(theta2)
    )
    return np.arccos(min(1.0, max(-1.0, dot)))

@jit(nopython=True)
def compute_row_events(i, state, orientation_theta, orientation_phi, T,
                       atom_type, defects, nu, nu_dep, E_b, E_diff, kT, T_melt, I0, delta_T_c,
                       top_layer, L, states_w, states_re, states_c, impurity_c):
    events = []
    row_state = state[i]
    row_theta = orientation_theta[i]
    row_phi = orientation_phi[i]
    row_T = T[i]
    row_atom_type = atom_type[i]
    row_defects = defects[i]

    # Deposition on top layer with impurity probability
    if i == top_layer:
        empty_top = row_state == 0  # Empty = 0
        for j, k in zip(*np.where(empty_top)):
            r = np.random.random()
            if r < impurity_c and impurity_c > 0:
                atom = 3  # C = 3
            elif r < impurity_c + IMPURITY_RE:
                atom = 2  # Re = 2
            else:
                atom = 1  # W = 1
            events.append(('dep', (i, j, k), nu_dep, None, atom))

    # Loop over occupied sites
    for j, k in zip(*np.where(row_state != 0)):  # Empty = 0
        if row_state[j, k] == 4:  # Defect = 4
            continue

        atom = row_state[j, k]
        if atom not in E_b or atom not in E_diff:
            continue

        neighbors = get_bcc_neighbors(i, j, k, L)
        n_bonds = 0
        for ni, nj, nk in neighbors:
            if state[ni, nj, nk] != 0:  # Empty = 0
                n_bonds += 1

        defect_factor = 1.5 if row_defects[j, k] == 1 else 1.0

        # Detachment
        E_det = n_bonds * E_b[atom]
        rate_det = nu * np.exp(-defect_factor * E_det / (kT * row_T[j, k]))
        if rate_det > RATE_THRESHOLD:
            events.append(('det', (i, j, k), rate_det, None, atom))

        # Evaporation
        if n_bonds < 4:
            E_evap = (8 - n_bonds) * E_b[atom]
            rate_evap = nu * np.exp(-defect_factor * E_evap / (kT * row_T[j, k]))
            if rate_evap > RATE_THRESHOLD:
                events.append(('evap', (i, j, k), rate_evap, None, atom))

        # Diffusion
        for ni, nj, nk in neighbors:
            if state[ni, nj, nk] == 0:  # Empty = 0
                E_diff_atom = E_diff[atom] + (n_bonds * E_b[atom] * 0.1)
                rate_diff = nu * np.exp(-defect_factor * E_diff_atom / (kT * row_T[j, k]))
                if rate_diff > RATE_THRESHOLD:
                    events.append(('diff', (i, j, k), rate_diff, (ni, nj, nk), atom))

    # Loop over empty sites for nucleation & attachment
    for j, k in zip(*np.where(row_state == 0)):  # Empty = 0
        delta_T = T_melt - row_T[j, k]
        if delta_T > delta_T_c:
            neighbors = get_bcc_neighbors(i, j, k, L)
            n_impurities = 0
            for ni, nj, nk in neighbors:
                if state[ni, nj, nk] in [2, 3]:  # Re = 2, C = 3
                    n_impurities += 1
            nuc_boost = 1.0 + 10.0 * n_impurities
            rate_nuc = I0 * nuc_boost * np.exp(-CARBON_SOLUTION_ENERGY / (kT * row_T[j, k]))
            atom_nuc = states_w  # W = 1
            if rate_nuc > RATE_THRESHOLD:
                events.append(('nuc', (i, j, k), rate_nuc, None, atom_nuc))

        for ni, nj, nk in neighbors:
            if state[ni, nj, nk] != 0:  # Empty = 0
                neighbor_atom = state[ni, nj, nk]
                orientation = np.array([
                    np.sin(row_theta[nj, nk]) * np.cos(row_phi[nj, nk]),
                    np.sin(row_theta[nj, nk]) * np.sin(row_phi[nj, nk]),
                    np.cos(row_theta[nj, nk])
                ])

                grad_x = (T[min(i+1, L-1), j, k] - T[max(i-1, 0), j, k]) / 2
                grad_y = (T[i, min(j+1, L-1), k] - T[i, max(j-1, 0), k]) / 2
                grad_z = (T[i, j, min(k+1, L-1)] - T[i, j, max(k-1, 0)]) / 2
                gradient = np.array([grad_x, grad_y, grad_z])
                norm = np.linalg.norm(gradient)
                if norm > EPSILON:
                    gradient /= norm
                else:
                    gradient = np.array([0.0, 0.0, 1.0])

                dot_product = np.dot(orientation, gradient)
                cos_theta = dot_product if np.linalg.norm(orientation) * norm > EPSILON else 1.0
                cos_mis = compute_misorientation(
                    orientation_theta[i, j, k], orientation_phi[i, j, k],
                    orientation_theta[ni, nj, nk], orientation_phi[ni, nj, nk]
                )

                E_att = E_b[neighbor_atom] * (1 - cos_mis)
                rate_att = (
                    nu * np.exp(-max(0, E_att) / (kT * row_T[j, k]))
                    * max(0, cos_theta) * ANISOTROPY_FACTOR
                )
                if rate_att > RATE_THRESHOLD:
                    events.append(('att', (i, j, k), rate_att, (ni, nj, nk), neighbor_atom))

    return events

def get_event_rates(state, orientation_theta, orientation_phi, T, atom_type, defects, L, states_w, states_re, states_c, step=0, debug_step=1000, impurity_c=0.0):
    E_b = np.array([0.0, E_B_W, E_B_RE, E_B_C, 0.0], dtype=np.float64)
    E_diff = np.array([0.0, E_DIFF_W, E_DIFF_RE, E_DIFF_C, 0.0], dtype=np.float64)
    top_layer = L - 1
    events = []
    for i in range(L):
        row_events = compute_row_events(
            i, state, orientation_theta, orientation_phi, T,
            atom_type, defects, NU, NU_DEP, E_b, E_diff, K_T, T_MELT, 1e10, 100,
            top_layer, L, states_w, states_re, states_c, impurity_c  # Fixed: Pass impurity_c
        )
        for event in row_events:
            events.append(event)
        # Log possible events at debug steps
        if step % debug_step == 0 and step > 0:
            debug_events(row_events, step, i)
    return events

def debug_events(events, step, layer):
    """Log possible events for a given step and layer to a file."""
    with open(f"events_step_{step}_layer_{layer}.txt", 'w') as f:
        f.write(f"Possible events at step {step}, layer {layer}:\n")
        for event in events[:100]:  # Limit to 100 events for brevity
            event_type, pos, rate, target, atom = event
            f.write(f"Type: {event_type}, Pos: {pos}, Rate: {rate}, Target: {target}, Atom: {atom}\n")

def test_event_rates(events, filename="event_rates.txt"):
    with open(filename, 'w') as f:
        for event in events[:100]:
            f.write(str(event) + '\n')