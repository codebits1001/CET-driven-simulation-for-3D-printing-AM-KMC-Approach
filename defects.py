import numpy as np
from constants import DEFECT_PROB_BASE, CARBON_MIGRATION_ENERGY, K_T, T_SUB

def track_defects(state, atom_type, L, T=None):
    if L == 0:
        return np.zeros((0, 0, 0), dtype=int)
    defects = np.zeros((L, L, L), dtype=int)
    c_sites = (atom_type == 3)  # Hardcoded STATES['C']
    if np.any(c_sites):
        if T is not None:
            T_vals = T[c_sites]
            with np.errstate(divide='ignore', invalid='ignore'):
                valid_T = np.where(T_vals > 0, T_vals, T_SUB)  # Default to T_SUB
                prob = DEFECT_PROB_BASE * np.exp(-0.3 / (K_T * valid_T))  # Lowered to 0.3 eV
        else:
            prob = np.full(np.sum(c_sites), DEFECT_PROB_BASE)
        prob = np.clip(prob, 0.0, 1.0)  # Ensure valid probability
        defects[c_sites] = (np.random.random(np.sum(c_sites)) < prob).astype(int)
    return defects

def get_defect_density(defects, voxel_size=5e-6):
    volume = defects.size * (voxel_size ** 3)  # [m^3]
    return np.sum(defects) / volume if volume > 0 else 0.0

def introduce_defects(state, atom_type, T=None, apply_to_state=False, voxel_size=5e-6):
    L = state.shape[0]
    mask = track_defects(state, atom_type, L, T)
    if apply_to_state:
        state[mask == 1] = 4  # Hardcoded STATES['Defect']
    density = get_defect_density(mask, voxel_size)
    return mask, density