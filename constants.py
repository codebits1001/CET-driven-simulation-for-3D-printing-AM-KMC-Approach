
# AFter implementing defect taracking. Test_Exp: 2(20000steps).

"""
Constants for KMC-based CET Simulation in Additive Manufacturing
All constants explicitly documented with physical units.
"""

# ------------------------------
# Identifiers / State Mapping
# ------------------------------
STATES = {
    'Empty': 0,
    'W': 1,
    'Re': 2,
    'C': 3,
    'Defect': 4,
}
DEFECT_ID = STATES['Defect']    # used when defects are written into atom_type/state

COLORS = {
    STATES['Empty']: (1.0, 1.0, 1.0),
    STATES['W']:     (0.2, 0.6, 1.0),
    STATES['Re']:    (0.8, 0.5, 0.2),
    STATES['C']:     (0.1, 0.8, 0.1),
    STATES['Defect']:(1.0, 0.0, 0.0),
}

# ------------------------------
# Simulation Parameters
# ------------------------------
LATTICE_SIZE = 30              # [voxels] per edge, cubic lattice
VOXEL_SIZE   = 5e-6             # [m] linear dimension of one voxel
N_STEPS      = 20000            # [steps] long run for validation
METRIC_UPDATE_STEP = 200         # [steps]
VISUAL_UPDATE_STEP = 2000        # [steps]
N_SEEDS = 20                    # [count]

# ------------------------------
# Physical Constants (for W system)
# ------------------------------
K_T   = 8.617333262e-5          # [eV/K] Boltzmann constant
T_MELT = 3695                   # [K] Tungsten melting point
T_SUB  = 2800                   # [K] substrate temperature (baseline)

ATOMIC_SPACING_W = 2.74e-10     # [m] W nearest-neighbor spacing

# Attempt frequencies (order-of-magnitude)
NU       = 1e13                 # [1/s] generic vibrational frequency
NU_DEP   =  2e13                 # [1/s] deposition attempt frequency

# Binding / diffusion (coarse, phenomenological)
E_B_W    = 3.8                  # [eV]
E_DIFF_W = 0.35                 # [eV]

# ------------------------------
# Impurities (Re and C)
# ------------------------------
E_B_RE      = 4.2               # [eV]
E_DIFF_RE   = 0.50              # [eV]
IMPURITY_RE = 0.10              # [-] default fraction (used if enabled)

E_B_C                 = 3.2     # [eV]
E_DIFF_C              = 0.30    # [eV]
CARBON_SOLUTION_ENERGY = 0.25   # [eV]
CARBON_MIGRATION_ENERGY = 0.15  # [eV]
IMPURITY_C            = 0.20    # [-] default fraction; you sweep 0.0/0.1/0.2

MAX_IMP_FRACTION = 1.0          # [-] upper bound guard

# ------------------------------
# Growth / Transport Scales
# ------------------------------
# Thermal gradient (1D estimate across the build height of LATTICE_SIZE*VOXEL_SIZE)
G = (T_MELT - T_SUB) / (LATTICE_SIZE * VOXEL_SIZE)   # [K/m]

# Deposition speed in voxel and SI units
R_VOX = (NU_DEP * ATOMIC_SPACING_W) / VOXEL_SIZE     # [voxels/s]
R_SI  = NU_DEP * ATOMIC_SPACING_W                    # [m/s]

ANISOTROPY_FACTOR = 0.25      # [-] small directional bias (used by your rate model)

# ------------------------------
# CET Classification Thresholds
# ------------------------------
CET_EQ_THRESHOLD = 0.50       # [-] equiaxed fraction cutoff
CET_AR_THRESHOLD = 3.0        # [-] aspect-ratio cutoff
CET_GR_THRESHOLD = 5e6        # [K·s/m²] threshold on G/R_phys (tuned down from 1e7)
CRITICAL_GRAIN_DENSITY = 1e5  # [1/m²] coarse cutoff for nucleation-rich regimes
CET_CHECK_INTERVAL = 100      # [steps]
PLOT_INTERVAL      = 500      # [steps]

# ------------------------------
# Event-Rate Prefactors (demo kinetics)
# ------------------------------
NU_EVAP          = 1e3        # [1/s]
NU_DIFF          = 5e2        # [1/s]
NU_NUCLEATION    = 5e-3       # [1/s]
EVAP_ACT_ENERGY  = 0.70       # [eV]
DIFF_ACT_ENERGY  = 0.40       # [eV]
NUCLEATION_ENERGY= 0.80       # [eV]
DELTA_T_C = 10               # [K]  # <--- RE-ADD THIS

# Local sticking / nucleation modifiers
E_STICK   = 0.10              # [eV] weak barrier for sticking
S0_STICK  = 0.50              # [-] baseline sticking coefficient
DELTA_T_REF = 50.0            # [K] reference undercooling for nucleation boost
I0          = 5e13            # [1/s] nucleation prefactor
K_NUC       = 500             # [eV·K^2] temperature penalty
BETA_IMP_NUC= 0.4             # [per-fraction] impurity-assisted nucleation

# ------------------------------
# Defect Parameters (coupled to your `defect.py`)
# ------------------------------
DEFECT_FORMATION_RATE = 1e-4   # [-] only if used elsewhere
DEFECT_PROB           = 3e-3   # [-] legacy toggle used by some modules
DEFECT_PROB_BASE      = 0.12   # [-] baseline in `track_defects` (yields nonzero densities)
DEFECT_DEP_REDUCTION  = 0.8   # [-] optional rate modifier near defects
DEFECT_EVAP_BOOST     = 2.5    # [-] optional evaporation boost near defects

# ------------------------------
# Numerics / Guards
# ------------------------------
RATE_THRESHOLD = 1e-30 # for 30          # [1/s] guard for empty-event termination
EPSILON        = 1e-10         # [-]
RANDOM_SEED    = 42            # [-]
