import numpy as np
from numba import jit
from constants import LATTICE_SIZE, N_SEEDS, STATES, T_SUB, IMPURITY_RE, IMPURITY_C
from kmc_event_rates import get_bcc_neighbors, compute_misorientation
from defects import introduce_defects

@jit(nopython=True)
def dfs_cluster(state, visited, cluster_label, orientation_theta, theta_threshold=0.3):
    """Depth-first search to identify clusters based on orientation similarity."""
    L = state.shape[0]
    stack = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if state[i, j, k] != 0 and visited[i, j, k] == 0:  
                    cluster = []
                    stack.append((i, j, k))
                    visited[i, j, k] = cluster_label
                    cluster.append((i, j, k))
                    while stack:
                        ci, cj, ck = stack.pop()
                        neighbors = get_bcc_neighbors(ci, cj, ck, L)
                        for ni, nj, nk in neighbors:
                            if state[ni, nj, nk] != 0 and visited[ni, nj, nk] == 0:
                                misorientation = compute_misorientation(
                                    orientation_theta[ci, cj, ck], 0,
                                    orientation_theta[ni, nj, nk], 0
                                )
                                if misorientation < theta_threshold:
                                    visited[ni, nj, nk] = cluster_label
                                    cluster.append((ni, nj, nk))
                                    stack.append((ni, nj, nk))
                    if cluster:
                        yield cluster
                    cluster_label += 1
    return cluster_label

def get_clusters(state, orientation_theta, theta_threshold=0.3):
    """Identify all clusters in the lattice."""
    L = state.shape[0]
    visited = np.zeros((L, L, L), dtype=np.int32)
    clusters = []
    cluster_label = 1
    for cluster in dfs_cluster(state, visited, cluster_label, orientation_theta, theta_threshold):
        clusters.append(cluster)
        cluster_label += 1
    return clusters, visited

def compute_metrics(clusters, state, atom_type):
    """Compute microstructure metrics."""
    L = state.shape[0]
    sizes = [len(cluster) for cluster in clusters if cluster]
    
    if not sizes:
        return 0.0, 0.0, [], {1: 0, 2: 0, 3: 0}
    
    grain_size = np.mean(sizes)
    max_size = max(sizes)
    min_size = min(sizes)
    aspect_ratio = max_size / max(min_size, 1)
    coverage = np.sum(state != 0) / (L * L * L)
    
    impurity_counts = {
        1: int(np.sum(atom_type == 1)),  # W
        2: int(np.sum(atom_type == 2)),  # Re
        3: int(np.sum(atom_type == 3))   # C
    }
    return aspect_ratio, coverage, sizes, impurity_counts

def get_cluster_edges(visited):
    """Calculate cluster edges."""
    L = visited.shape[0]
    edges = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if visited[i, j, k] != 0:
                    for ni, nj, nk in get_bcc_neighbors(i, j, k, L):
                        if visited[ni, nj, nk] != visited[i, j, k]:
                            edges.append(((i, j, k), (ni, nj, nk)))
    return edges

def initialize_state():
    """Initialize the lattice with seeds, impurities, and defects."""
    np.random.seed(42)

    state = np.zeros((LATTICE_SIZE, LATTICE_SIZE, LATTICE_SIZE), dtype=int)
    orientation_theta = np.zeros((LATTICE_SIZE, LATTICE_SIZE, LATTICE_SIZE))
    orientation_phi = np.zeros((LATTICE_SIZE, LATTICE_SIZE, LATTICE_SIZE))
    atom_type = np.zeros((LATTICE_SIZE, LATTICE_SIZE, LATTICE_SIZE), dtype=int)
    T = np.full((LATTICE_SIZE, LATTICE_SIZE, LATTICE_SIZE), T_SUB, dtype=float)

    # Place seeds at bottom layer
    bottom_layer_size = LATTICE_SIZE * LATTICE_SIZE
    seed_indices = np.random.choice(bottom_layer_size, size=N_SEEDS, replace=False)
    seed_x, seed_y = np.unravel_index(seed_indices, (LATTICE_SIZE, LATTICE_SIZE))
    state[seed_x, seed_y, 0] = STATES['W']
    atom_type[seed_x, seed_y, 0] = STATES['W']
    orientation_theta[seed_x, seed_y, 0] = np.random.uniform(0, np.pi, size=N_SEEDS)
    orientation_phi[seed_x, seed_y, 0] = np.random.uniform(0, 2*np.pi, size=N_SEEDS)

    # Add impurities
    non_empty = state != STATES['Empty']
    num_non_empty = np.sum(non_empty)
    if num_non_empty > 0:
        re_mask = np.random.random(num_non_empty) < IMPURITY_RE
        c_mask = np.random.random(num_non_empty) < IMPURITY_C
        atom_type[non_empty] = np.where(re_mask, STATES['Re'], STATES['W'])
        atom_type[non_empty] = np.where(c_mask, STATES['C'], atom_type[non_empty])
        state[non_empty] = atom_type[non_empty]

    # Add defects
    introduce_defects(state, atom_type)

    return state, orientation_theta, orientation_phi, T, atom_type