# metrics.py
import numpy as np
from constants import VOXEL_SIZE, CET_AR_THRESHOLD, CET_EQ_THRESHOLD
from utils import get_clusters, calculate_aspect_ratio

def grain_aspect_ratio(state, voxel_size=VOXEL_SIZE):
    """Compute aspect ratio of occupied region (height vs lateral)."""
    occupied = np.argwhere(state > 0)
    if occupied.size == 0:
        return 0.0
    z_extent = occupied[:, 2].max() - occupied[:, 2].min() + 1
    x_extent = occupied[:, 0].max() - occupied[:, 0].min() + 1
    y_extent = occupied[:, 1].max() - occupied[:, 1].min() + 1
    lateral = max(x_extent, y_extent)
    return (z_extent * voxel_size) / (lateral * voxel_size + 1e-12)  # dimensionless

def equiaxed_fraction(state, threshold=CET_AR_THRESHOLD, voxel_size=VOXEL_SIZE):
    """Fraction of grains with AR below threshold → equiaxed fraction (dimensionless)."""
    occupied = np.argwhere(state > 0)
    if occupied.size == 0:
        return 0.0
    ar = grain_aspect_ratio(state, voxel_size)
    return 1.0 if ar < threshold else 0.0

def nucleation_density(state, voxel_size=VOXEL_SIZE):
    """Count nucleation sites (occupied cells) normalized by volume [1/m³]."""
    occupied = np.count_nonzero(state > 0)
    volume = np.prod(state.shape) * (voxel_size**3)
    return occupied / volume if volume > 0 else 0.0

def grain_sizes(state):
    """Compute approximate grain sizes as connected components (placeholder)."""
    occupied = np.argwhere(state > 0)
    if occupied.size == 0:
        return []
    return [len(occupied)]  # treat all occupied as one cluster

import numpy as np
from utils import get_clusters, calculate_aspect_ratio
from constants import VOXEL_SIZE, CET_AR_THRESHOLD, CET_EQ_THRESHOLD
def compute_metrics(state, theta, phi, defects=None, voxel_size=VOXEL_SIZE,
                    W_mask=None, Re_mask=None, C_mask=None, grain_ids=None, rng_seed=None):
    clusters, cluster_sizes = get_clusters(state, theta, phi, theta_threshold=0.5)
    if not clusters:
        return {
            "AspectRatio": 0.0, "EquiaxedFraction": 0.0, "NucleationDensity": 0.0,
            "AvgGrainSize": 0.0, "GrainCount": 0, "DefectDensity": 0.0,
            "Frac_W": 0.0, "Frac_Re": 0.0, "Frac_C": 0.0,
            "C_boundary_frac": 0.0, "Re_boundary_frac": 0.0,
            "Defect_voxel_count": 0, "Defect_voxel_frac": 0.0,
            "Grain_d50_um": 0.0, "Grain_d90_um": 0.0,
            "VOXEL_SIZE_m": voxel_size, "RANDOM_SEED": rng_seed
        }

    # --- existing metrics ---
    aspect_ratios = [calculate_aspect_ratio(c) for c in clusters]
    avg_ar = np.mean(aspect_ratios) if aspect_ratios else 0.0
    eq_frac = np.mean(np.array(aspect_ratios) < CET_AR_THRESHOLD) if aspect_ratios else 0.0
    volume = state.size * (voxel_size ** 3)
    nuc_dens = len(clusters) / volume if volume > 0 else 0.0
    avg_size = np.mean([len(c) for c in clusters]) * voxel_size * 1e6  # µm
    def_count = np.sum(defects) if defects is not None else 0
    def_dens = def_count / volume if volume > 0 else 0.0

    # --- new fractions ---
    total_voxels = state.size
    frac_w = compute_voxel_fraction(np.count_nonzero(W_mask), total_voxels) if W_mask is not None else 0.0
    frac_re = compute_voxel_fraction(np.count_nonzero(Re_mask), total_voxels) if Re_mask is not None else 0.0
    frac_c = compute_voxel_fraction(np.count_nonzero(C_mask), total_voxels) if C_mask is not None else 0.0

    # --- boundary fractions ---
    c_boundary_frac = compute_boundary_fraction(C_mask, grain_ids) if (C_mask is not None and grain_ids is not None) else 0.0
    re_boundary_frac = compute_boundary_fraction(Re_mask, grain_ids) if (Re_mask is not None and grain_ids is not None) else 0.0

    # --- grain equivalent diameters ---
    d50, d90 = equivalent_diameter_um(cluster_sizes, voxel_size)

    return {
        "AspectRatio": avg_ar,
        "EquiaxedFraction": eq_frac,
        "NucleationDensity": nuc_dens,
        "AvgGrainSize": avg_size,
        "GrainCount": len(clusters),
        "DefectDensity": def_dens,
        "Frac_W": frac_w,
        "Frac_Re": frac_re,
        "Frac_C": frac_c,
        "C_boundary_frac": c_boundary_frac,
        "Re_boundary_frac": re_boundary_frac,
        "Defect_voxel_count": def_count,
        "Defect_voxel_frac": compute_voxel_fraction(def_count, total_voxels),
        "Grain_d50_um": d50,
        "Grain_d90_um": d90,
        "VOXEL_SIZE_m": voxel_size,
        "RANDOM_SEED": rng_seed
    }

    
def compute_CET(state, theta, phi, voxel_size=VOXEL_SIZE):
    m = compute_metrics(state, theta, phi, voxel_size=voxel_size)
    return "Equiaxed" if m["AspectRatio"] < CET_AR_THRESHOLD and m["EquiaxedFraction"] > CET_EQ_THRESHOLD else "Columnar"

def detect_CET_transition(metrics_dict):
    return (metrics_dict["AspectRatio"] < CET_AR_THRESHOLD and
            metrics_dict["EquiaxedFraction"] > CET_EQ_THRESHOLD)



def compute_voxel_fraction(count, total_voxels):
    """Return fraction of voxels of a given type."""
    return count / total_voxels if total_voxels > 0 else 0.0

def compute_boundary_fraction(impurity_mask, grain_ids):
    """
    Compute fraction of impurity atoms sitting on grain boundaries.
    A voxel is boundary if any 6-neighbor has a different grain ID.
    """
    boundary_count = 0
    total_impurities = np.count_nonzero(impurity_mask)

    # Pad arrays for safe neighbor check
    padded = np.pad(grain_ids, 1, mode='edge')
    imp_pad = np.pad(impurity_mask, 1, mode='constant')

    for x in range(1, grain_ids.shape[0] + 1):
        for y in range(1, grain_ids.shape[1] + 1):
            for z in range(1, grain_ids.shape[2] + 1):
                if imp_pad[x, y, z]:
                    gid = padded[x, y, z]
                    neighs = [
                        padded[x+1,y,z], padded[x-1,y,z],
                        padded[x,y+1,z], padded[x,y-1,z],
                        padded[x,y,z+1], padded[x,y,z-1]
                    ]
                    if any(n != gid for n in neighs):
                        boundary_count += 1

    return boundary_count / total_impurities if total_impurities > 0 else 0.0


def equivalent_diameter_um(cluster_sizes, voxel_size):
    """
    Compute grain equivalent diameters from voxel cluster sizes.
    Return median (d50) and 90th percentile (d90) in microns.
    """
    if len(cluster_sizes) == 0:
        return 0.0, 0.0
    volumes = np.array(cluster_sizes) * (voxel_size ** 3)
    diameters = ( (6.0 * volumes / np.pi) ** (1.0/3.0) ) * 1e6  # → µm
    return np.median(diameters), np.percentile(diameters, 90)
