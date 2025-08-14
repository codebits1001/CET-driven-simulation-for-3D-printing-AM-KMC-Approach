import numpy as np
from constants import (
    LATTICE_SIZE, N_SEEDS, STATES, T_SUB,
    IMPURITY_RE, IMPURITY_C,
    CET_GR_THRESHOLD, CET_AR_THRESHOLD, NU_DEP
)
from kmc_event_rates import get_bcc_neighbors, compute_misorientation
from defects import introduce_defects

# =============================
# CLUSTER DETECTION
# =============================
def get_cluster_edges(cluster):
    edges = set()
    coords_set = set(tuple(int(x) for x in voxel) for voxel in cluster)
    for voxel in coords_set:
        i, j, k = voxel
        neighbors = [
            (i+1, j, k), (i-1, j, k),
            (i, j+1, k), (i, j-1, k),
            (i, j, k+1), (i, j, k-1)
        ]
        for n in neighbors:
            if n in coords_set:
                edges.add(tuple(sorted([voxel, n])))
    return list(edges)

def dfs_cluster(state, visited, cluster_label, orientation_theta, 
               orientation_phi=None, theta_threshold=0.5):
    """Depth-first search for grain clustering with orientation checking"""
    Lx, Ly, Lz = state.shape  # Get all three dimensions
    stack = []
    
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                if state[i,j,k] != 0 and visited[i,j,k] == 0:
                    cluster = []
                    stack.append((i,j,k))
                    visited[i,j,k] = cluster_label
                    cluster.append((i,j,k))
                    
                    while stack:
                        ci, cj, ck = stack.pop()
                        neighbors = get_bcc_neighbors(ci, cj, ck, Lx)  # Use Lx for lattice size
                        
                        for ni, nj, nk in neighbors:
                            if (0 <= ni < Lx and 0 <= nj < Ly and 0 <= nk < Lz and
                                state[ni,nj,nk] != 0 and visited[ni,nj,nk] == 0):
                                
                                if orientation_phi is None:
                                    misorientation = abs(orientation_theta[ci,cj,ck] - orientation_theta[ni,nj,nk])
                                else:
                                    misorientation = compute_misorientation(
                                        orientation_theta[ci,cj,ck], orientation_phi[ci,cj,ck],
                                        orientation_theta[ni,nj,nk], orientation_phi[ni,nj,nk]
                                    )
                                
                                if misorientation < theta_threshold:
                                    visited[ni,nj,nk] = cluster_label
                                    cluster.append((ni,nj,nk))
                                    stack.append((ni,nj,nk))
                    
                    if cluster:
                        yield cluster
                    cluster_label += 1
    return cluster_label

def get_clusters(state, orientation_theta, orientation_phi=None, theta_threshold=0.5):
    """Get all grain clusters in the lattice"""
    if state.size == 0:
        return [], np.array([])
        
    Lx, Ly, Lz = state.shape
    visited = np.zeros((Lx, Ly, Lz), dtype=np.int32)
    clusters = []
    cluster_label = 1
    
    for cluster in dfs_cluster(state, visited, cluster_label, 
                             orientation_theta, orientation_phi, theta_threshold):
        clusters.append([tuple(map(int, voxel)) for voxel in cluster])
        cluster_label += 1
        
    return clusters, visited
# =============================
# MICROSTRUCTURE METRICS
# =============================
def compute_metrics(clusters, state, atom_type):
    L = state.shape[0]
    sizes = [len(cluster) for cluster in clusters if cluster]
    if not sizes:
        return 0.0, 0.0, [], {STATES['W']: 0, STATES['Re']: 0, STATES['C']: 0}
    max_size = max(sizes)
    min_size = min(sizes)
    aspect_ratio = max_size / max(min_size, 1)
    coverage = np.sum(state != STATES['Empty']) / float(L * L * L)
    impurity_counts = {
        STATES['W']: int(np.sum(atom_type == STATES['W'])),
        STATES['Re']: int(np.sum(atom_type == STATES['Re'])),
        STATES['C']: int(np.sum(atom_type == STATES['C'])),
    }
    return aspect_ratio, coverage, sizes, impurity_counts

def calculate_aspect_ratio(cluster):
    coords = np.array(cluster)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims = maxs - mins + 1
    longest = np.max(dims)
    shortest = np.min(dims)
    return float(longest) / float(max(shortest, 1))

# =============================
# CET METRICS
# =============================

def detect_CET_transition(state, orientation_theta, threshold=0.3):
    """Detect CET by measuring equiaxed fraction in top layers"""
    L = state.shape[2]  # Get z-dimension size
    top_layer_start = max(0, L - int(L*threshold))  # Ensure we don't go negative
    top_layers = state[:, :, top_layer_start:]  # Get top layers
    
    # Get clusters in top layers only
    clusters, _ = get_clusters(top_layers, orientation_theta[:, :, top_layer_start:])
    
    if not clusters:
        return False
    
    equiaxed_count = 0
    for cluster in clusters:
        coords = np.array(cluster)
        if len(coords) == 0:
            continue
            
        # Convert relative z-coords back to absolute
        z_coords = coords[:, 2] + top_layer_start
        z_range = np.max(z_coords) - np.min(z_coords)
        xy_span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
        
        # Equiaxed criterion (width > 1.5*height)
        if xy_span > z_range * 1.5:
            equiaxed_count += 1
    
    return equiaxed_count / len(clusters) > 0.5  # >50% equiaxed grains

def calculate_G_over_R(G, R):
    if R == 0:
        return np.inf
    return float(G) / float(R)

def determine_CET(aspect_ratio, G_over_R, gr_threshold=CET_GR_THRESHOLD, ar_threshold=CET_AR_THRESHOLD):
    if G_over_R is None or np.isinf(G_over_R):
        print(f"Debug: G_over_R invalid, using AspectRatio={aspect_ratio:.2f} vs threshold={ar_threshold}")
        return "Columnar" if aspect_ratio >= ar_threshold else "Equiaxed"
    print(f"Debug: G_over_R={G_over_R:.2e}, AspectRatio={aspect_ratio:.2f}, thresholds={gr_threshold}/{ar_threshold}")
    return "Columnar" if (G_over_R >= gr_threshold and aspect_ratio >= ar_threshold) else "Equiaxed"

def validate_aspect_ratio(aspect_ratio, ar_threshold=CET_AR_THRESHOLD):
    return aspect_ratio >= ar_threshold

def validate_CET_with_GR(G, R, aspect_ratio, gr_threshold=CET_GR_THRESHOLD, ar_threshold=CET_AR_THRESHOLD):
    G_over_R = calculate_G_over_R(G, R)
    return determine_CET(aspect_ratio, G_over_R, gr_threshold, ar_threshold)

def overall_microstructure_classification(clusters, G=None, R=None,
                                        gr_threshold=CET_GR_THRESHOLD, ar_threshold=CET_AR_THRESHOLD):
    if not clusters:
        return "No grains detected"
    aspect_ratios = [calculate_aspect_ratio(c) for c in clusters]
    avg_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
    if G is None or R is None:
        return determine_CET(avg_ar, None, gr_threshold, ar_threshold)
    return validate_CET_with_GR(G, R, avg_ar, gr_threshold, ar_threshold)

def compute_CET(state, orientation_theta, orientation_phi=None, G_over_R=None):
    if orientation_phi is None:
        orientation_phi = np.zeros_like(orientation_theta)  # Create zero array if None
    clusters, _ = get_clusters(state, orientation_theta, orientation_phi)
    if not clusters:
        print("Debug: No clusters detected, returning 'No grains detected'")
        return "No grains detected"
    aspect_ratios = [calculate_aspect_ratio(c) for c in clusters]
    avg_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
    return determine_CET(avg_ar, G_over_R)

# =============================
# THERMAL & GROWTH RATE HELPERS
# =============================
def estimate_temperature_gradient(T_field):
    dz = 1.0  # mm per voxel
    grad_z = np.abs(np.gradient(T_field, dz, axis=2))
    non_zero = grad_z > 0
    return float(np.mean(grad_z[non_zero])) if np.any(non_zero) else 0.0

def estimate_growth_rate(previous_state, current_state, timestep):
    dz = 1.0  # mm per voxel
    prev_idx = np.where(previous_state != STATES['Empty'])
    curr_idx = np.where(current_state != STATES['Empty'])
    prev_solid = int(prev_idx[2].max()) if prev_idx[0].size > 0 else 0
    curr_solid = int(curr_idx[2].max()) if curr_idx[0].size > 0 else 0
    if timestep <= 0 or curr_solid == prev_solid:
        return NU_DEP * 1e-16
    return float((curr_solid - prev_solid) * dz / timestep)

# =============================
# INITIALIZATION
# =============================
def initialize_state():
    np.random.seed(42)
    state = np.zeros((LATTICE_SIZE, LATTICE_SIZE, LATTICE_SIZE), dtype=int)
    orientation_theta = np.zeros_like(state, dtype=float)
    orientation_phi = np.zeros_like(state, dtype=float)
    atom_type = np.zeros_like(state, dtype=int)
    T = np.full_like(state, T_SUB, dtype=float)
    bottom_layer_size = LATTICE_SIZE * LATTICE_SIZE
    seed_indices = np.random.choice(bottom_layer_size, size=N_SEEDS, replace=False)
    seed_x, seed_y = np.unravel_index(seed_indices, (LATTICE_SIZE, LATTICE_SIZE))
    state[seed_x, seed_y, 0] = STATES['W']
    atom_type[seed_x, seed_y, 0] = STATES['W']
    orientation_theta[seed_x, seed_y, 0] = np.random.uniform(0, np.pi, size=N_SEEDS)
    orientation_phi[seed_x, seed_y, 0] = np.random.uniform(0, 2*np.pi, size=N_SEEDS)
    non_empty = state != STATES['Empty']
    num_non_empty = int(np.sum(non_empty))
    if num_non_empty > 0:
        re_mask = np.random.random(num_non_empty) < IMPURITY_RE
        c_mask = np.random.random(num_non_empty) < IMPURITY_C
        tmp = np.where(re_mask, STATES['Re'], STATES['W'])
        tmp = np.where(c_mask, STATES['C'], tmp)
        atom_type[non_empty] = tmp
        state[non_empty] = tmp
    introduce_defects(state, atom_type)
    return state, orientation_theta, orientation_phi, T, atom_type