"""
Optimized constants for KMC grain growth simulation with reliable CET behavior
"""

# -----------------------------
# Simulation Parameters
# -----------------------------
LATTICE_SIZE = 30
N_STEPS = 10000  # Increased for clearer CET development
METRIC_UPDATE_STEP = 500
VISUAL_UPDATE_STEP = 1000
N_SEEDS = 60  # Balanced between columnar and equiaxed nucleation

# -----------------------------
# Physical Constants (Tungsten W)
# -----------------------------
NU = 1e13
NU_DEP = 5e12  # Reduced for more controlled growth
E_B_W = 3.8  # Slightly reduced from 4.0 for more mobility
E_DIFF_W = 0.35  # Increased from 0.45 for faster diffusion
T_MELT = 3695  # Restored actual tungsten melting point
T_SUB = 2000  # Increased from 1800 for better grain boundary mobility

# -----------------------------
# Thermal Parameters
# -----------------------------
K_T = 8.617e-5
G = (T_MELT - T_SUB)/LATTICE_SIZE  # Thermal gradient (K/voxel)

# -----------------------------
# Impurity Constants
# -----------------------------
# Rhenium (Re)
E_B_RE = 4.2  # Reduced from 4.4
E_DIFF_RE = 0.5  # Increased from 0.55
IMPURITY_RE = 0.1  # Increased from 0.08

# Carbon (C)
E_B_C = 3.2  # Reduced from 3.4
E_DIFF_C = 0.3  # Increased from 0.35
CARBON_SOLUTION_ENERGY = 0.25  # Reduced from 0.3
CARBON_MIGRATION_ENERGY = 1.5  # Adjusted from 2.0
IMPURITY_C = 0.2  # Increased from 0.15

# -----------------------------
# Growth Dynamics
# -----------------------------
ANISOTROPY_FACTOR = 0.1  # Increased from 0.03 for stronger directional growth
R = NU_DEP * 1e-6  # Growth rate (voxels/s)

# -----------------------------
# CET Control Parameters
# -----------------------------
CET_GR_THRESHOLD = 1e5  # Changed from 0.4 (new units)
CET_AR_THRESHOLD = 3.0  # Increased from 2.0
CET_CHECK_INTERVAL = 100
PLOT_INTERVAL = 500

# -----------------------------
# Event Parameters
# -----------------------------
I0 = 5e13  # Increased from 1e13
DELTA_T_C = 20  # Reduced from 30

# -----------------------------
# Defect Parameters
# -----------------------------
DEFECT_PROB = 0.05  # Increased from 0.03
DEFECT_PROB_BASE = 0.3  # Balanced from 0.5

# -----------------------------
# State Definitions
# -----------------------------
STATES = {
    'Empty': 0,
    'W': 1,
    'Re': 2,
    'C': 3,
    'Defect': 4
}

COLORS = {
    STATES['Empty']: (1.0, 1.0, 1.0),
    STATES['W']: (0.2, 0.6, 1.0),
    STATES['Re']: (0.8, 0.5, 0.2),
    STATES['C']: (0.1, 0.8, 0.1),
    STATES['Defect']: (1.0, 0.0, 0.0)
}

# -----------------------------
# Numerical Parameters
# -----------------------------
RATE_THRESHOLD = 1e-20
EPSILON = 1e-10
RANDOM_SEED = 42