# # constants.py
# """
# Global constants for KMC grain growth simulation in BCC tungsten (W) with impurities.
# """

# # Simulation Parameters
# LATTICE_SIZE = 30             # Increased to capture larger grains
# N_STEPS = 20000              # Reduced to focus on CET transition
# METRIC_UPDATE_STEP =4000       # More frequent for detailed metrics
# VISUAL_UPDATE_STEP = 4000      # Frequent snapshots for CET visualization
# N_SEEDS = 30                   # Reduced to ~2.4% coverage for clearer columnar growth

# # Physical Constants (Tungsten W)
# NU = 1e13                      # Attempt frequency
# NU_DEP = 1e15                # Lowered deposition rate for slower growth
# E_B_W = 4.0                    # eV, W–W binding energy
# E_DIFF_W = 0.5                 # eV, W diffusion activation energy
# T_MELT = 3695                  # K, melting point of tungsten
# T_SUB = 1500                   # K, increased to reduce defect density

# # Boltzmann Constant
# K_T = 8.617e-5                 # eV/K

# # Impurity Constants (Rhenium Re)
# E_B_RE = 4.5                   # eV, W–Re binding energy
# E_DIFF_RE = 0.6               # eV, Re diffusion energy
# IMPURITY_RE = 0.05             # Reduced to 5% for controlled equiaxed nucleation

# # Impurity Constants (Carbon C)
# E_B_C = 3.5                   # eV, W–C binding energy
# E_DIFF_C = 0.4                 # eV, C diffusion energy
# CARBON_SOLUTION_ENERGY = 0.5  # eV
# CARBON_MIGRATION_ENERGY = 1.46 # eV
# IMPURITY_C = 0.1               # Reduced to 10% for gradual CET

# # Growth Anisotropy
# ANISOTROPY_FACTOR = 0.1        # Increased to strengthen columnar growth

# # State Identifiers
# STATES = {
#     'Empty': 0,
#     'W': 1,
#     'Re': 2,
#     'C': 3,
#     'Defect': 4
# }

# # Numerical Stability & Thresholds
# RATE_THRESHOLD = 1e-20
# DEFECT_PROB = 0.05              # Reduced to lower initial defects
# DEFECT_PROB_BASE = 0.1
# EPSILON = 1e-10

# # Visualization Colors (normalized RGB)
# COLORS = {
#     STATES['Empty']: (1.0, 1.0, 1.0),
#     STATES['W']: (0.2, 0.6, 1.0),
#     STATES['Re']: (0.8, 0.5, 0.2),
#     STATES['C']: (0.1, 0.8, 0.1),
#     STATES['Defect']: (1.0, 0.0, 0.0)
# }

# # Random Seed
# RANDOM_SEED = 42

#with cet parameters.
# constants.py
"""
Global constants for KMC grain growth simulation in BCC tungsten (W) with impurities.
This set is tuned to *promote CET* (Columnar-to-Equiaxed Transition) within 20,000 steps.
"""

# -----------------------------
# Simulation Parameters
# -----------------------------
LATTICE_SIZE = 40              # Larger domain → allows columnar and equiaxed zones to coexist
N_STEPS = 20000                 # Enough time for CET to develop
METRIC_UPDATE_STEP = 4000       # Capture evolution at several stages
VISUAL_UPDATE_STEP = 4000       # Snapshots to visualize CET onset
N_SEEDS = 80                    # Higher initial seeds → faster equiaxed nucleation

# -----------------------------
# Physical Constants (Tungsten W)
# -----------------------------
NU = 1e13                       # Attempt frequency (unchanged)
NU_DEP = 5e14                    # Slightly slower deposition rate → promotes lateral growth
E_B_W = 4.0                      # eV, W–W binding energy (unchanged)
E_DIFF_W = 0.45                  # Lower diffusion barrier → allows faster isotropic growth
T_MELT = 3695                    # K, tungsten melting point
T_SUB = 1800                     # Higher substrate temp → improves grain boundary mobility

# -----------------------------
# Boltzmann Constant
# -----------------------------
K_T = 8.617e-5                   # eV/K

# -----------------------------
# Impurity Constants (Rhenium Re)
# -----------------------------
E_B_RE = 4.4                     # Slightly less than pure columnar case → easier detachment
E_DIFF_RE = 0.55                 # Lower diffusion barrier → promotes spreading
IMPURITY_RE = 0.08               # Slightly higher → more heterogeneous nucleation sites

# -----------------------------
# Impurity Constants (Carbon C)
# -----------------------------
E_B_C = 3.4                      # Lower → C atoms less “locked”, enabling more mobility
E_DIFF_C = 0.35                  # Higher mobility to assist CET
CARBON_SOLUTION_ENERGY = 0.45    # eV
CARBON_MIGRATION_ENERGY = 1.4    # eV
IMPURITY_C = 0.15                # More C → enhanced equiaxed nucleation

# -----------------------------
# Growth Anisotropy
# -----------------------------
ANISOTROPY_FACTOR = 0.03         # Reduced anisotropy → weakens columnar dominance

# -----------------------------
# State Identifiers
# -----------------------------
STATES = {
    'Empty': 0,
    'W': 1,
    'Re': 2,
    'C': 3,
    'Defect': 4
}

# -----------------------------
# Numerical Stability & Thresholds
# -----------------------------
RATE_THRESHOLD = 1e-20
DEFECT_PROB = 0.03               # Lower → fewer early defects so CET isn’t masked
DEFECT_PROB_BASE = 0.08
EPSILON = 1e-10

# -----------------------------
# Visualization Colors (normalized RGB)
# -----------------------------
COLORS = {
    STATES['Empty']: (1.0, 1.0, 1.0),
    STATES['W']: (0.2, 0.6, 1.0),
    STATES['Re']: (0.8, 0.5, 0.2),
    STATES['C']: (0.1, 0.8, 0.1),
    STATES['Defect']: (1.0, 0.0, 0.0)
}

# -----------------------------
# Random Seed
# -----------------------------
RANDOM_SEED = 42
