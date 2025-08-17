import numpy as np
import time
import os
import pandas as pd

from kmc_simulation import run_kmc
from lattice_init import initialize_lattice, save_lattice
from visualization import visualize_final_state, visualize_grain_structure
from constants import (
    LATTICE_SIZE, N_STEPS, T_SUB, DEFECT_PROB, N_SEEDS,
    IMPURITY_C, METRIC_UPDATE_STEP, DEFECT_ID
)
from graphs import plot_combined_metrics
from metrics import detect_CET_transition


def main():
    """Main execution for CET study with varying carbon concentrations"""
    print("Starting KMC simulation for microstructure control...")
    start_time = time.time()

    # Define the impurity (carbon) levels to simulate
    carbon_levels = [0.0, 0.1, 0.2]

    # Store results for later comparison
    metrics_data = {
        'carbon_levels': [],
        'grain_sizes': [],
        'defect_densities': [],
        'aspect_ratios': []
    }

    for c_level in carbon_levels:
        prefix = f"impurity_c_{int(c_level*100)}"
        print(f"\nRunning simulation with {c_level*100:.1f}% carbon")

        # Create output directories
        output_dir = f"outputs/{prefix}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/microstructures", exist_ok=True)
        os.makedirs("output_images", exist_ok=True)

        # Initialize lattice
        state, theta, phi, T, atom_type = initialize_lattice(
            lattice_size=LATTICE_SIZE,
            n_seeds=N_SEEDS,
            T_sub=T_SUB,
            random_seed=42,
            impurity_c=c_level
        )
        save_lattice(state, theta, phi, T, atom_type,
                     prefix=f"{output_dir}/init")

        # Run KMC simulation
        run_start_time = time.time()
        state, atom_type, total_time, theta, phi = run_kmc(
            L=LATTICE_SIZE,
            n_steps=N_STEPS,
            temp=T_SUB,
            defect_fraction=DEFECT_PROB,
            n_seeds=N_SEEDS,
            impurity_c=c_level,
            output_prefix=prefix
        )
        run_end_time = time.time()

        # Read final metrics
        cet_status = "Undetected"
        csv_path = f"outputs/{prefix}/metrics.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                final_metrics = df.iloc[-1].to_dict()
                cet_status = "Equiaxed" if detect_CET_transition(final_metrics) else "Columnar"

                metrics_data['carbon_levels'].append(c_level)
                metrics_data['grain_sizes'].append(final_metrics.get("AvgGrainSize", np.nan))
                metrics_data['defect_densities'].append(final_metrics.get("DefectDensity", np.nan))  # ✅ now real
                metrics_data['aspect_ratios'].append(final_metrics.get("AspectRatio", np.nan))
        else:
            print(f"Metrics file not found at {csv_path}")

        # Save visualizations
        visualize_grain_structure(
            state, theta, phi, atom_type, N_STEPS, c_level,
            filename=f"{output_dir}/microstructures/final_grain.png"
        )
        visualize_final_state(
            state, atom_type, theta, phi,
            title=f"Final State: {cet_status} (C={c_level*100:.0f}%)",
            filename=f"output_images/final_state_{prefix}.png"
        )

        print(f"Completed {prefix} in {run_end_time-run_start_time:.2f}s")

    # Combined metrics comparison across carbon levels
    if len(metrics_data['carbon_levels']) == len(carbon_levels):
        plot_combined_metrics(
            metrics_data['carbon_levels'],
            metrics_data['grain_sizes'],
            metrics_data['defect_densities'],   # ✅ defects included
            metrics_data['aspect_ratios'],
            save_dir="outputs/combined_metrics"
        )
    else:
        print("Incomplete data for combined metrics plot")

    print(f"\nTotal runtime: {time.time()-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
