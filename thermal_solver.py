import numpy as np
from scipy.ndimage import laplace
from constants import LATTICE_SIZE, T_MELT, T_SUB, VOXEL_SIZE

# Material constants (Tungsten, units explicit)
K = 173.0        # [W/m-K] thermal conductivity
RHO = 19300.0    # [kg/m^3] density
CP = 132.0       # [J/kg-K] specific heat
ALPHA = K / (RHO * CP)  # [m^2/s] thermal diffusivity

# Laser parameters (simple Gaussian surface heating)
DEFAULT_BEAM_RADIUS = 50e-6  # [m] 50 µm spot
DEFAULT_ABSORPTIVITY = 0.35  # [dimensionless] absorptivity at ~1 µm wavelength

def build_temperature_field(L: int = None) -> np.ndarray:
    """
    Steady vertical gradient field, consistent with CET baseline studies.
    Axis convention:
      i = vertical (0 bottom/substrate, L-1 top/hot)
      j, k = in-plane axes
    T(i) = T_SUB + ((T_MELT - T_SUB)/(L-1)) * i
    """
    if L is None:
        L = LATTICE_SIZE

    i = np.arange(L, dtype=np.float64)
    if L > 1:
        G = (T_MELT - T_SUB) / (L - 1)  # [K per voxel]
    else:
        G = 0.0
    T_i = T_SUB + G * i
    T = np.repeat(T_i[:, None, None], L, axis=1)
    T = np.repeat(T, L, axis=2)
    return T

def update_temperature(
    T: np.ndarray,
    state: np.ndarray,
    prev_state: np.ndarray,
    dt: float,
    laser_pos: tuple,
    laser_power: float,
    beam_radius: float = DEFAULT_BEAM_RADIUS,
    absorptivity: float = DEFAULT_ABSORPTIVITY,
) -> np.ndarray:
    """
    Explicit Euler update for temperature with:
      - 3D 6-neighbor Laplacian (zero-flux boundaries, vectorized)
      - Gaussian surface laser source at i == L-1
      - Latent heat release on solidification (state changes 0 -> nonzero)

    Parameters
    ----------
    T : (L,L,L) float64
      Temperature field [K]
    state, prev_state : (L,L,L) int
      Lattice occupancy; solidification detected by prev= l0 and curr!=0
    dt : float
      Timestep [s]
    laser_pos : (i0, j0) in voxel indices (on the top surface i=L-1)
      Beam center position (vertical axis is i)
    laser_power : float
      Laser power [W]
    beam_radius : float
      e^{-1} Gaussian radius [m]
    absorptivity : float
      Surface absorptivity [0..1]

    Returns
    -------
    new_T : (L,L,L) float64
      Updated temperature, clipped to [T_SUB, 1.1*T_MELT]
    """
    L = T.shape[0]
    assert T.shape == (L, L, L)
    assert state.shape == T.shape and prev_state.shape == T.shape

    # Precompute constants
    inv_dx2 = 1.0 / (VOXEL_SIZE * VOXEL_SIZE)  # [1/m^2]

    # Laser surface heat flux at i == L-1 (top)
    i0, j0 = laser_pos
    jj = np.arange(L, dtype=np.float64)
    kk = np.arange(L, dtype=np.float64)
    JJ, KK = np.meshgrid(jj, kk, indexing='ij')
    r_m = np.sqrt((JJ - j0) ** 2 + (KK - j0) ** 2) * VOXEL_SIZE  # [m]
    area_norm = np.pi * beam_radius * beam_radius
    I_surface = (laser_power * absorptivity / area_norm) * np.exp(-(r_m ** 2) / (beam_radius ** 2))  # [W/m^2]

    # Vectorized Laplacian
    lap = laplace(T) * inv_dx2  # [K/m^2]

    # Laser volumetric source (top layer only)
    q_vol = np.zeros_like(T)
    q_vol[-1, :, :] = I_surface / VOXEL_SIZE  # [W/m^3]

    # Latent heat: detect solidification (prev empty -> now filled)
    solidification_mask = (prev_state == 0) & (state != 0)
    dF_dt = solidification_mask.astype(float) / max(dt, 1e-12)  # [1/s]

    # Explicit Euler: dT/dt = alpha * Laplacian + laser + latent heat
    dTdt = ALPHA * lap + q_vol / (RHO * CP) + (200e3 / CP) * dF_dt  # [K/s]
    new_T = T + dt * dTdt

    return np.clip(new_T, T_SUB, T_MELT * 1.1)

def update_temperature_cet(T: np.ndarray, state: np.ndarray, dt: float = 1e-6) -> np.ndarray:
    """
    Simplified temperature update for CET baseline runs (no laser, steady gradient).
    Uses state as prev_state (no latent heat tracking).
    """
    L = T.shape[0]
    # Minimal diffusion to smooth numerical noise
    inv_dx2 = 1.0 / (VOXEL_SIZE * VOXEL_SIZE)
    lap = laplace(T) * inv_dx2
    new_T = T + dt * ALPHA * lap
    return np.clip(new_T, T_SUB, T_MELT * 1.1)