import numpy as np
import random
import pandas as pd
import os
from thermal_solver import update_temperature
from visualization import plot_CET_diagnostic
from kmc_event_rates import get_event_rates
from defects import introduce_defects
from utils import compute_metrics, get_clusters, compute_CET, detect_CET_transition
from constants import (
    LATTICE_SIZE, N_STEPS, RATE_THRESHOLD, METRIC_UPDATE_STEP, STATES,
    CET_CHECK_INTERVAL, T_SUB, T_MELT, NU_DEP, IMPURITY_C
)
from lattice_init import initialize_lattice

def run_kmc(L, n_steps, temp, defect_fraction, n_seeds, impurity_c,
            output_prefix="default"):
    """
    Final integrated KMC simulation with thermal dynamics and CET detection.
    
    Parameters:
    -----------
    L : int                     - Lattice size
    n_steps : int               - Number of KMC steps
    temp : float                - Substrate temperature (K)
    defect_fraction : float     - Initial defect fraction
    n_seeds : int               - Number of nucleation seeds
    impurity_c : float          - Carbon impurity concentration
    output_prefix : str         - Output file prefix
    
    Returns:
    --------
    tuple: (state, atom_type, total_time, orientation_theta, orientation_phi)
    """
    # Initialize system parameters
    laser_pos = np.array([L//2, L//2, L-1])  # Laser at top center
    laser_speed = 0.05  # Slower scanning speed
    thermal_update_interval = 20  # Balance accuracy and performance
    
    # Create output directories
    output_dir = f'outputs/{output_prefix}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/microstructures', exist_ok=True)

    # Initialize lattice
    state, orientation_theta, orientation_phi, T, atom_type = initialize_lattice(
        lattice_size=L, n_seeds=n_seeds, T_sub=temp, impurity_c=impurity_c
    )
    defects = introduce_defects(state, atom_type)
    
    # Growth rate parameters
    G = (T_MELT - T_SUB) / L  # Thermal gradient (K/unit length)
    R = NU_DEP * 1e-6        # Increased growth rate (was 1e-16)
    
    # Simulation tracking
    total_time = 0.0
    metrics_data = []
    nucleation_count = 0
    cet_detected = False

    for step in range(n_steps):
        # Update laser position (simple raster pattern)
        if step % 100 == 0:
            laser_pos[0] = (laser_pos[0] + int(laser_speed*L)) % L
            laser_pos[1] = (laser_pos[1] + int(laser_speed*L/2)) % L
        
        # Thermal update
        if step % thermal_update_interval == 0:
            T = update_temperature(T, state, total_time*thermal_update_interval, L, laser_pos)
        
        # Event handling
        events = get_event_rates(
            state, orientation_theta, orientation_phi, T, atom_type, defects, L,
            STATES['W'], STATES['Re'], STATES['C'], step, 1000, impurity_c
        )
        total_rate = sum(e[2] for e in events) if events else 0.0

        # Termination conditions
        if not events or total_rate <= 0 or total_rate < RATE_THRESHOLD:
            print(f"Termination at step {step}: No valid events (rate={total_rate:.2e})")
            break
            
        # Event selection
        r = random.random() * total_rate
        cumulative = 0.0
        for e in events:
            cumulative += e[2]
            if cumulative >= r:
                etype, pos, rate, target, atom = e
                i, j, k = pos
                
                # Execute event
                if etype == 'dep':
                    state[i,j,k] = atom
                    atom_type[i,j,k] = atom
                    orientation_theta[i,j,k] = np.random.uniform(0, np.pi)
                    orientation_phi[i,j,k] = np.random.uniform(0, 2*np.pi)
                    
                elif etype in ('det', 'evap'):
                    state[i,j,k] = STATES['Empty']
                    atom_type[i,j,k] = STATES['Empty']
                    orientation_theta[i,j,k] = 0.0
                    orientation_phi[i,j,k] = 0.0
                    
                elif etype == 'diff' and target:
                    ti, tj, tk = target
                    state[ti,tj,tk] = state[i,j,k]
                    atom_type[ti,tj,tk] = atom_type[i,j,k]
                    orientation_theta[ti,tj,tk] = orientation_theta[i,j,k]
                    orientation_phi[ti,tj,tk] = orientation_phi[i,j,k]
                    state[i,j,k] = STATES['Empty']
                    atom_type[i,j,k] = STATES['Empty']
                    orientation_theta[i,j,k] = 0.0
                    orientation_phi[i,j,k] = 0.0
                    
                elif etype == 'nuc':
                    state[i,j,k] = atom
                    atom_type[i,j,k] = atom
                    orientation_theta[i,j,k] = np.random.uniform(0, np.pi)
                    orientation_phi[i,j,k] = np.random.uniform(0, 2*np.pi)
                    nucleation_count += 1
                    
                elif etype == 'att' and target:
                    ni, nj, nk = target
                    state[i,j,k] = atom
                    atom_type[i,j,k] = atom
                    orientation_theta[i,j,k] = orientation_theta[ni,nj,nk]
                    orientation_phi[i,j,k] = orientation_phi[ni,nj,nk]
                
                break

        # Time update
        dt = max(-np.log(random.random()) / total_rate, 1e-12)
        total_time += dt
        
        # Periodic analysis
        if step % METRIC_UPDATE_STEP == 0 or step == n_steps - 1:
            # Update defects
            defects = introduce_defects(state, atom_type, T)
            
            # CET detection
            if not cet_detected and detect_CET_transition(state, orientation_theta):
                cet_detected = True
                print(f"CET detected at step {step} (G/R={G/R:.2e})")
                plot_CET_diagnostic(state, f"CET Detected (G/R={G/R:.2e})", 
                                  step, f"{output_dir}/cet_{step}.png")
            
            # Metrics collection
            clusters, _ = get_clusters(state, orientation_theta, orientation_phi)
            aspect_ratio, coverage, sizes, impurity_counts = compute_metrics(
                clusters, state, atom_type
            )
            
            metrics_data.append({
                'Step': step,
                'Time': total_time,
                'Coverage': coverage,
                'AspectRatio': aspect_ratio,
                'AvgGrainSize': np.mean(sizes) if sizes else 0.0,
                'DefectDensity': np.sum(defects)/defects.size if defects.size > 0 else 0.0,
                'W_Count': impurity_counts.get(STATES['W'], 0),
                'Re_Count': impurity_counts.get(STATES['Re'], 0),
                'C_Count': impurity_counts.get(STATES['C'], 0),
                'NucleationCount': nucleation_count,
                'G_over_R': G/R if R > 0 else np.inf,
                'CET': compute_CET(state, orientation_theta, orientation_phi, G/R),
                'CET_Detected': cet_detected
            })

            print(f"Step {step}: Size={np.mean(sizes):.1f}±{np.std(sizes):.1f}, "
                  f"AR={aspect_ratio:.2f}, CET={cet_detected}, Time={total_time:.2e}s")
    
    # Save metrics
    if metrics_data:
        output_path = f'{output_dir}/metrics.csv'
        pd.DataFrame(metrics_data).to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")

    print(f"Completed {step+1} steps in {total_time:.2e} s")
    return state, atom_type, total_time, orientation_theta, orientation_phi