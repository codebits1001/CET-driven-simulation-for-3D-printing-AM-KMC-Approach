import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from constants import (
    LATTICE_SIZE, N_SEEDS, T_SUB, T_MELT,
    STATES, IMPURITY_RE
)
from defects import introduce_defects

def initialize_lattice(lattice_size=LATTICE_SIZE, n_seeds=N_SEEDS,
                       T_sub=T_SUB, T_melt=T_MELT, random_seed=42, impurity_c=0.0):
    """
    Initialize the simulation lattice with random seeds for multiple materials.

    Parameters
    ----------
    lattice_size : int
        Size of the cubic lattice.
    n_seeds : int
        Number of nucleation sites to initialize.
    T_sub : float
        Substrate temperature.
    T_melt : float
        Melting temperature.
    random_seed : int
        Seed for reproducibility.
    impurity_c : float
        Carbon impurity fraction (passed from main.py).

    Returns
    -------
    state : ndarray (L, L, L)
        Lattice state IDs.
    orientation_theta : ndarray
        Crystal orientation theta angles.
    orientation_phi : ndarray
        Crystal orientation phi angles.
    T : ndarray
        Temperature field.
    atom_type : ndarray
        Type of atom at each site.
    """
    np.random.seed(random_seed)

    L = lattice_size
    state = np.full((L, L, L), STATES['Empty'], dtype=int)
    orientation_theta = np.zeros((L, L, L))
    orientation_phi = np.zeros((L, L, L))
    atom_type = np.full((L, L, L), STATES['Empty'], dtype=int)

    G = (T_melt - T_sub) / L
    z = np.arange(L)[np.newaxis, np.newaxis, :]
    T = np.full((L, L, L), T_sub, dtype=float) + G * z

    seed_indices = np.random.choice(L * L, n_seeds, replace=False)
    seed_x = seed_indices // L
    seed_y = seed_indices % L

    print(f"Initializing {n_seeds} seeds with IMPURITY_C={impurity_c}, IMPURITY_RE={IMPURITY_RE}")

    for idx, (x, y) in enumerate(zip(seed_x, seed_y)):
        rand = np.random.random()
        if rand < IMPURITY_RE:
            atom = STATES['Re']
        elif impurity_c > 0 and rand < (IMPURITY_RE + impurity_c):
            atom = STATES['C']
        else:
            atom = STATES['W']
        print("Seed", idx, "at (", x, ",", y, ", 0): rand =", rand, ", atom =", atom)
        state[x, y, 0] = atom
        atom_type[x, y, 0] = atom
        orientation_theta[x, y, 0] = np.random.uniform(0, np.pi)
        orientation_phi[x, y, 0] = np.random.uniform(0, 2 * np.pi)

    

    return state, orientation_theta, orientation_phi, T, atom_type

def visualize_initial_seeds(state, atom_type,
                            title="Initial Nucleation Sites",
                            filename="initial_seeds.png"):
    """3D scatter plot of initial nucleation sites."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        STATES['W']: 'grey',
        STATES['Re']: 'blue',
        STATES['C']: 'red',
        STATES['Defect']: 'black'
    }
    labels = {
        STATES['W']: 'Tungsten',
        STATES['Re']: 'Rhenium',
        STATES['C']: 'Carbon',
        STATES['Defect']: 'Defect'
    }

    for atom, color in colors.items():
        sites = np.where(state == atom)
        if len(sites[0]) > 0:
            ax.scatter(
                sites[2], sites[1], sites[0],
                c=color, label=labels[atom], alpha=0.6, s=10
            )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Build Direction)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def save_lattice(state, orientation_theta, orientation_phi, T, atom_type,
                 prefix="init"):
    """Save lattice state and associated fields to .npy files."""
    np.save(f'{prefix}_state.npy', state)
    np.save(f'{prefix}_orientation_theta.npy', orientation_theta)
    np.save(f'{prefix}_orientation_phi.npy', orientation_phi)
    np.save(f'{prefix}_temperature.npy', T)
    np.save(f'{prefix}_atom_type.npy', atom_type)