import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import os
from constants import LATTICE_SIZE, STATES, COLORS
from utils import get_clusters, get_cluster_edges

def plot_lattice(state, title="Lattice State", step=None, filename=None, impurity_c=0.0):
    """Plot a single slice of the 3D lattice (middle layer) and save to file."""
    L = state.shape[0]
    mid_z = L // 2
    slice_data = state[:, :, mid_z]

    state_values = list(COLORS.keys())
    color_list = [COLORS[state] for state in state_values]
    state_names = [name for name, value in STATES.items() if value in state_values]
    cmap = ListedColormap(color_list)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(slice_data.T, origin='lower', cmap=cmap, interpolation='none')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[state], label=name)
        for state, name in zip(state_values, state_names) if state in COLORS
    ]
    plt.legend(handles=legend_elements, title="Atom Types", loc='upper right')
    
    plt.title(f"{title}, Impurity C={impurity_c*100:.1f}%")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()

    os.makedirs('output_images', exist_ok=True)
    if filename is None:
        filename = f"output_images/lattice_midZ_step_{step}_impurity_c_{int(impurity_c*100)}.png" if step is not None else f"output_images/lattice_midZ_impurity_c_{int(impurity_c*100)}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

def visualize_layer(state, z, step, impurity_c):
    x, y = np.where(state[:, :, z] != STATES['Empty'])
    if len(x) == 0:
        return
    atom_types = state[x, y, z]
    colors = np.array([
        'grey' if a == STATES['W'] else
        'blue' if a == STATES['Re'] else
        'red' if a == STATES['C'] else
        'black' if a == STATES['Defect'] else
        'purple'
        for a in atom_types
    ])
    plt.figure(figsize=(6, 6))
    plt.scatter(y, x, c=colors, s=50, alpha=0.7)
    plt.title(f'Layer z={z} at Step {step}, Impurity C={impurity_c*100:.1f}%')
    plt.xlabel('Y')
    plt.ylabel('X')
    os.makedirs('output_images', exist_ok=True)
    plt.savefig(f'output_images/layer_z{z}_step_{step}_impurity_c_{int(impurity_c*100)}.png', dpi=100)
    plt.close()

def visualize_grain_structure(state, orientation_theta, orientation_phi, atom_type, step, impurity_c, filename=None):
    clusters, _ = get_clusters(state, orientation_theta, orientation_phi)
    if not clusters:
        return
    
    cluster_dict = {i: cluster for i, cluster in enumerate(clusters)}
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = colormaps['tab20']
    
    for cluster_id, points in cluster_dict.items():
        coords = np.array(points)
        if len(coords) == 0:
            continue
        ax.scatter(coords[:, 1], coords[:, 0], coords[:, 2], 
                   c=[cmap(cluster_id % 20)], s=50, alpha=0.7)
    
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Grain Structure, Impurity C={impurity_c*100:.1f}%')
    os.makedirs('output_images', exist_ok=True)
    if filename is None:
        filename = f'output_images/final_grain_structure_impurity_c_{int(impurity_c*100)}.png'
    plt.savefig(filename, dpi=150)
    plt.close()

def visualize_final_state(state, atom_type, orientation_theta=None, orientation_phi=None, impurity_c=0.0,
                          title="Final Lattice State", filename=None):
    """
    3D scatter plot of final lattice state with atom-specific colors;
    optionally draws grain boundaries for CET visualization.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for atom_id, color in COLORS.items():
        if atom_id == STATES['Empty']:
            continue
        sites = np.where(state == atom_id)
        if len(sites[0]) > 0:
            ax.scatter(
                sites[2], sites[1], sites[0],
                c=[color],
                label=list(STATES.keys())[list(STATES.values()).index(atom_id)],
                alpha=0.6, s=10
            )

    if orientation_theta is not None and orientation_phi is not None:
        clusters, visited = get_clusters(state, orientation_theta, orientation_phi)
        for cluster in clusters:
            edges = get_cluster_edges(cluster)
            for (x1, y1, z1), (x2, y2, z2) in edges:
                ax.plot(
                    [z1, z2], [y1, y2], [x1, x2],
                    'k-', linewidth=0.5, alpha=0.3
                )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Build Direction)')
    plt.title(f"{title}, Impurity C={impurity_c*100:.1f}%")
    plt.legend()
    plt.tight_layout()

    os.makedirs('output_images', exist_ok=True)
    if filename is None:
        filename = f"output_images/final_state_impurity_c_{int(impurity_c*100)}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_CET_diagnostic(state, cet_status, step, filename):
    """Visualize CET transition with status overlay"""
    plt.figure(figsize=(10, 8))
    mid_z = state.shape[0] // 2
    plt.imshow(state[:, :, mid_z].T, cmap='viridis', origin='lower')
    plt.title(f"Step {step}: {cet_status}")
    plt.colorbar(label='Grain ID')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(filename, dpi=150)
    plt.close()