# thermal_solver.py
import numpy as np
from constants import LATTICE_SIZE, T_MELT, T_SUB

def update_temperature(T, state, dt, lattice_size, laser_pos=None):
    """
    Update temperature field with:
    - Heat diffusion
    - Laser heat source (if position provided)
    - Natural cooling
    """
    L = lattice_size
    new_T = np.copy(T)
    dx = 1.0  # Lattice spacing
    
    # Simplified thermal model
    for i in range(1, L-1):
        for j in range(1, L-1):
            for k in range(1, L-1):
                # Heat diffusion term
                laplacian = (T[i+1,j,k] + T[i-1,j,k] +
                            T[i,j+1,k] + T[i,j-1,k] +
                            T[i,j,k+1] + T[i,j,k-1] - 6*T[i,j,k]) / dx**2
                
                # Laser heat source (top surface only)
                Q = 0
                if k == L-1 and laser_pos is not None:
                    r_sq = (i-laser_pos[0])**2 + (j-laser_pos[1])**2
                    Q = 100 * np.exp(-r_sq/(2*(L/4)**2))  # Gaussian heat source
                
                new_T[i,j,k] = T[i,j,k] + dt * (0.01 * laplacian + Q - 0.001*(T[i,j,k]-T_SUB))
    
    # Boundary conditions
    new_T[:,:,0] = T_SUB  # Constant substrate temperature
    return np.clip(new_T, T_SUB, T_MELT*1.1)