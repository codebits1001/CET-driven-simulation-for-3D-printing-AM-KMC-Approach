import numpy as np
from numba import jit
from constants import (
    NU, NU_DEP,
    E_B_W, E_B_RE, E_B_C,
    E_DIFF_W, E_DIFF_RE, E_DIFF_C,
    K_T, T_MELT,
    RATE_THRESHOLD, ANISOTROPY_FACTOR, EPSILON, IMPURITY_RE,
    CARBON_SOLUTION_ENERGY, I0, DELTA_T_C, STATES
)

@jit(nopython=True, cache=True)
def get_bcc_neighbors(i, j, k, L):
    """Enhanced BCC neighbor calculation with boundary checks"""
    neighbors = []
    # 1st nearest neighbors (8)
    for di, dj, dk in [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                       (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
        ni, nj, nk = i+di, j+dj, k+dk
        if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
            neighbors.append((ni, nj, nk))
    # 2nd nearest neighbors (6)
    for di, dj, dk in [(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2)]:
        ni, nj, nk = i+di, j+dj, k+dk
        if 0 <= ni < L and 0 <= nj < L and 0 <= nk < L:
            neighbors.append((ni, nj, nk))
    return neighbors

@jit(nopython=True, cache=True)
def compute_misorientation(theta1, phi1, theta2, phi2):
    """Optimized misorientation calculation with better numerical stability"""
    # Vector components
    v1 = np.array([
        np.sin(theta1)*np.cos(phi1),
        np.sin(theta1)*np.sin(phi1),
        np.cos(theta1)
    ])
    v2 = np.array([
        np.sin(theta2)*np.cos(phi2),
        np.sin(theta2)*np.sin(phi2),
        np.cos(theta2)
    ])
    
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    dot = max(min(dot, 1.0), -1.0)  # Clamp to valid range
    return np.arccos(dot)

@jit(nopython=True)
def compute_row_events(i, state, orientation_theta, orientation_phi, T,
                     atom_type, defects, nu, nu_dep, E_b, E_diff, kT, T_melt,
                     I0, delta_T_c, top_layer, L, states_w, states_re, states_c,
                     impurity_c):
    """Numba-compatible event calculation with thermal and defect effects"""
    events = []
    row_state = state[i]
    row_theta = orientation_theta[i]
    row_phi = orientation_phi[i]
    row_T = T[i]
    row_defects = defects[i]

    # --------------------------------
    # Deposition with thermal effects
    # --------------------------------
    if i == top_layer:
        empty_indices = np.where(row_state == 0)
        for idx in range(len(empty_indices[0])):
            j = empty_indices[0][idx]
            k = empty_indices[1][idx]
            
            # Temperature-dependent deposition rate
            thermal_factor = np.exp(-(T_melt - row_T[j,k])/(kT*row_T[j,k]))
            effective_nu_dep = nu_dep * thermal_factor
            
            r = np.random.random()
            if r < impurity_c:
                atom = states_c
            elif r < impurity_c + IMPURITY_RE:
                atom = states_re
            else:
                atom = states_w
                
            events.append(('dep', (i,j,k), effective_nu_dep, None, atom))

    # --------------------------------
    # Occupied site events
    # --------------------------------
    occ_indices = np.where(row_state != 0)
    for idx in range(len(occ_indices[0])):
        j = occ_indices[0][idx]
        k = occ_indices[1][idx]
        
        atom = row_state[j,k]
        if atom == 4:  # Skip defects
            continue
            
        # Material parameters
        if atom == states_w:
            E_b_atom = E_b[0]
            E_diff_atom = E_diff[0]
        elif atom == states_re:
            E_b_atom = E_b[1]
            E_diff_atom = E_diff[1]
        elif atom == states_c:
            E_b_atom = E_b[2]
            E_diff_atom = E_diff[2]
        else:
            continue

        # Count bonds
        neighbors = get_bcc_neighbors(i,j,k,L)
        n_bonds = 0
        for ni, nj, nk in neighbors:
            if state[ni,nj,nk] != 0:
                n_bonds += 1

        defect_factor = 1.5 if row_defects[j,k] else 1.0
        local_temp = row_T[j,k]

        # Detachment
        E_det = n_bonds * E_b_atom
        rate_det = nu * np.exp(-defect_factor*E_det/(kT*local_temp))
        if rate_det > RATE_THRESHOLD:
            events.append(('det', (i,j,k), rate_det, None, atom))

        # Evaporation
        if n_bonds < 4:
            E_evap = (8 - n_bonds) * E_b_atom
            rate_evap = nu * np.exp(-defect_factor*E_evap/(kT*local_temp))
            if rate_evap > RATE_THRESHOLD:
                events.append(('evap', (i,j,k), rate_evap, None, atom))

        # Diffusion
        for ni, nj, nk in neighbors:
            if state[ni,nj,nk] == 0:  # Empty neighbor
                delta_T = abs(row_T[j,k] - T[ni,nj,nk])
                grad_factor = 1.0 + 0.1*delta_T/(T_melt - T[ni,nj,nk] + EPSILON)
                
                E_diff_total = E_diff_atom + (n_bonds * E_b_atom * 0.1)
                rate_diff = nu * grad_factor * np.exp(-defect_factor*E_diff_total/(kT*local_temp))
                if rate_diff > RATE_THRESHOLD:
                    events.append(('diff', (i,j,k), rate_diff, (ni,nj,nk), atom))

    # --------------------------------
    # Empty site events
    # --------------------------------
    empty_indices = np.where(row_state == 0)
    for idx in range(len(empty_indices[0])):
        j = empty_indices[0][idx]
        k = empty_indices[1][idx]
        
        local_temp = row_T[j,k]
        delta_T = T_melt - local_temp
        
        # Nucleation
        if delta_T > delta_T_c:
            neighbors = get_bcc_neighbors(i,j,k,L)
            n_impurities = 0
            for ni, nj, nk in neighbors:
                if state[ni,nj,nk] == states_re or state[ni,nj,nk] == states_c:
                    n_impurities += 1
            
            nuc_boost = 1.0 + 10.0 * n_impurities * (delta_T/delta_T_c)**2
            rate_nuc = I0 * nuc_boost * np.exp(-CARBON_SOLUTION_ENERGY/(kT*local_temp))
            if rate_nuc > RATE_THRESHOLD:
                events.append(('nuc', (i,j,k), rate_nuc, None, states_w))

        # Attachment
        neighbors = get_bcc_neighbors(i,j,k,L)
        for ni, nj, nk in neighbors:
            if state[ni,nj,nk] != 0:  # Occupied neighbor
                neighbor_atom = state[ni,nj,nk]
                
                # Misorientation
                mis_ang = compute_misorientation(
                    row_theta[j,k], row_phi[j,k],
                    orientation_theta[ni,nj,nk], orientation_phi[ni,nj,nk]
                )
                
                # Thermal gradient
                grad_z = (T[i,j,min(k+1,L-1)] - T[i,j,max(k-1,0)]) / 2.0
                grad_factor = max(0.0, grad_z) / (T_melt - T[ni,nj,nk] + EPSILON)
                
                E_att = E_b[neighbor_atom] * (1.0 - np.cos(mis_ang))
                rate_att = nu * np.exp(-E_att/(kT*local_temp)) * (1.0 + ANISOTROPY_FACTOR * grad_factor)
                if rate_att > RATE_THRESHOLD:
                    events.append(('att', (i,j,k), rate_att, (ni,nj,nk), neighbor_atom))

    return events

def get_event_rates(state, orientation_theta, orientation_phi, T, atom_type, defects, L,
                   states_w, states_re, states_c, step=0, debug_step=1000, impurity_c=0.0):
    """Main event rate calculation with thermal integration"""
    # Parameter arrays for numba
    E_b = np.array([E_B_W, E_B_RE, E_B_C])
    E_diff = np.array([E_DIFF_W, E_DIFF_RE, E_DIFF_C])
    
    top_layer = L - 1
    events = []
    
    # Parallelize by layer if needed (remove for numba compatibility)
    for i in range(L):
        row_events = compute_row_events(
            i, state, orientation_theta, orientation_phi, T,
            atom_type, defects, NU, NU_DEP, E_b, E_diff, K_T, T_MELT,
            I0, DELTA_T_C, top_layer, L, states_w, states_re, states_c, impurity_c
        )
        events.extend(row_events)

    return events