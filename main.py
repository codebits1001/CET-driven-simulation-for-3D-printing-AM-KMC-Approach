import numpy as np
import time
import os
import pandas as pd
from kmc_simulation import run_kmc
from lattice_init import initialize_lattice, save_lattice
from visualization import visualize_final_state, visualize_grain_structure
from constants import (
    LATTICE_SIZE, N_STEPS, T_SUB, DEFECT_PROB, N_SEEDS, 
    T_MELT, NU_DEP, STATES, IMPURITY_C, METRIC_UPDATE_STEP
)
from graphs import plot_combined_metrics
from utils import get_clusters, detect_CET_transition

def main():
    """Main execution for CET study with varying carbon concentrations"""
    print("Starting KMC simulation for microstructure control...")
    start_time = time.time()

    carbon_levels = [0.0, 0.1, 0.2]
    metrics_data = {
        'carbon_levels': [],
        'grain_sizes': [], 
        'defect_densities': [], 
        'aspect_ratios': []
    }

    for c_level in carbon_levels:
        prefix = f"impurity_c_{int(c_level*100)}"
        print(f"\nRunning simulation with {c_level*100:.1f}% carbon")
        
        # Setup directories
        output_dir = f"outputs/{prefix}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/microstructures", exist_ok=True)
        os.makedirs("output_images", exist_ok=True)

        # Initialize system
        state, orientation_theta, orientation_phi, T, atom_type = initialize_lattice(
            lattice_size=LATTICE_SIZE,
            n_seeds=N_SEEDS,
            T_sub=T_SUB,
            random_seed=42,
            impurity_c=c_level
        )
        save_lattice(state, orientation_theta, orientation_phi, T, atom_type, 
                    prefix=f"{output_dir}/init")

        # Run simulation
        run_start_time = time.time()
        state, atom_type, total_time, orientation_theta, orientation_phi = run_kmc(
            L=LATTICE_SIZE,
            n_steps=N_STEPS,
            temp=T_SUB,
            defect_fraction=DEFECT_PROB,
            n_seeds=N_SEEDS,
            impurity_c=c_level,
            output_prefix=prefix
        )
        run_end_time = time.time()

        # Post-simulation
        cet_status = "Equiaxed" if detect_CET_transition(state, orientation_theta) else "Columnar"
        
        # Visualization
        visualize_grain_structure(
            state, orientation_theta, atom_type, N_STEPS, c_level,
            filename=f"{output_dir}/microstructures/final_grain.png"
        )
        visualize_final_state(
            state, atom_type, orientation_theta,
            title=f"Final State: {cet_status} (C={c_level*100}%)",
            filename=f"output_images/final_state_{prefix}.png"
        )

        # Metrics collection
        csv_path = f"{output_dir}/metrics.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                step_data = df[df['Step'] == METRIC_UPDATE_STEP]
                if not step_data.empty:
                    metrics_data['carbon_levels'].append(c_level)
                    metrics_data['grain_sizes'].append(step_data['AvgGrainSize'].iloc[0])
                    metrics_data['defect_densities'].append(step_data['DefectDensity'].iloc[0])
                    metrics_data['aspect_ratios'].append(step_data['AspectRatio'].iloc[0])
            except Exception as e:
                print(f"Error processing {csv_path}: {str(e)}")
        else:
            print(f"Metrics file not found at {csv_path}")

        print(f"Completed {prefix} in {run_end_time-run_start_time:.2f}s")

    # Final analysis
    if len(metrics_data['carbon_levels']) == len(carbon_levels):
        plot_combined_metrics(
            metrics_data['carbon_levels'],
            metrics_data['grain_sizes'],
            metrics_data['defect_densities'],
            metrics_data['aspect_ratios'],
            save_dir="outputs/combined_metrics"
        )
    else:
        print("Incomplete data for combined metrics plot")

    print(f"\nTotal runtime: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()