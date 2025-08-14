import numpy as np
from constants import *

def track_defects(state, atom_type, L, T=None):
    """
    Track defects with optional temperature-dependent probability.

    Parameters
    ----------
    state : np.ndarray
        3D lattice state array.
    atom_type : np.ndarray
        3D array with atom types (matches STATES IDs).
    L : int
        Lattice size (assumes cubic lattice).
    T : np.ndarray or None
        Temperature array per site (same shape as state), optional.

    Returns
    -------
    np.ndarray
        Binary array (1 = defect, 0 = no defect) same shape as state.
    """
    if L == 0:
        return np.zeros((0, 0, 0), dtype=int)

    defects = np.zeros((L, L, L), dtype=int)
    c_sites = (atom_type == STATES['C'])

    if np.any(c_sites):
        if T is not None:
            # Temperature-dependent defect probability using Arrhenius relation
            T_vals = T[c_sites]
            with np.errstate(divide='ignore', invalid='ignore'):
                valid_T = np.where(T_vals > 0, T_vals, np.inf)
                defect_prob = DEFECT_PROB_BASE * np.exp(
                    -CARBON_MIGRATION_ENERGY / (K_T * valid_T)
                )
        else:
            defect_prob = np.full(np.sum(c_sites), DEFECT_PROB_BASE)

        # Assign defects where probability condition is met
        defects[c_sites] = (np.random.random(np.sum(c_sites)) < defect_prob).astype(int)

    return defects


def get_defect_density(defects):
    """
    Calculate defect density (fraction of sites that are defects).
    """
    if defects.size == 0:
        return 0.0
    return np.sum(defects) / defects.size

def introduce_defects(state, atom_type, T=None):
    """
    Introduce defects into the state array, modifying it in place.
    """
    L = state.shape[0]
    defects_mask = track_defects(state, atom_type, L, T) == 1
    state[defects_mask] = STATES['Defect']
    return defects_mask