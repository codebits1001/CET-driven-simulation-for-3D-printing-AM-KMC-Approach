# import numpy as np
# import time
# import os
# import pandas as pd
# from kmc_simulation import run_kmc
# from lattice_init import initialize_lattice, visualize_initial_seeds, save_lattice
# from visualization import visualize_final_state, visualize_grain_structure, visualize_possible_events
# from graphs import plot_combined_metrics
# from defects import introduce_defects
# from kmc_event_rates import get_event_rates
# from constants import LATTICE_SIZE, N_STEPS, T_SUB, DEFECT_PROB, N_SEEDS, VISUAL_UPDATE_STEP, T_MELT, NU_DEP, STATES, IMPURITY_C

# def main():
#     """
#     Main script to run KMC simulation for microstructure control in AM (CET study).
#     Runs multiple simulations with varying carbon impurity levels to study CET.
#     """
#     print("Starting KMC simulation for microstructure control...")
#     start_time = time.time()

#     carbon_levels = [0.0, 0.1, 0.2]  # 0%, 10%, 20%
#     metrics_at_500 = {'grain_sizes': [], 'defect_densities': [], 'aspect_ratios': []}

#     for c_level in carbon_levels:
#         print(f"\nRunning simulation with IMPURITY_C = {c_level*100:.1f}%")
#         prefix = f"impurity_c_{int(c_level*100)}"
#         os.makedirs(f"outputs/metrics/{prefix}", exist_ok=True)
#         os.makedirs(f"outputs/microstructures/{prefix}", exist_ok=True)
#         os.makedirs("output_images", exist_ok=True)

#         state, orientation_theta, orientation_phi, T, atom_type = initialize_lattice(
#             lattice_size=LATTICE_SIZE,
#             n_seeds=N_SEEDS,
#             T_sub=T_SUB,
#             random_seed=42,
#             impurity_c=c_level
#         )

#         visualize_initial_seeds(state, atom_type, filename=f"output_images/initial_seeds_{prefix}.png")
#         save_lattice(state, orientation_theta, orientation_phi, T, atom_type, prefix=f"init_{prefix}")
#         print(f"Initial lattice setup complete for {prefix}. Saved to 'init_{prefix}_*.npy' and 'initial_seeds_{prefix}.png'.")

#         G = (T_MELT - T_SUB) / LATTICE_SIZE
#         R = NU_DEP * 1e-16
#         gr_ratio = G / R if R > 0 else float('inf')
#         print(f"G/R ratio: {gr_ratio:.2e} K·s/unit (lower values promote equiaxed)")

#         run_start_time = time.time()
#         state, atom_type, total_time, orientation_theta, orientation_phi = run_kmc(
#             L=LATTICE_SIZE,
#             n_steps=N_STEPS,
#             temp=T_SUB,
#             defect_fraction=DEFECT_PROB,
#             n_seeds=N_SEEDS,
#             impurity_c=c_level,  # Fixed: Use c_level instead of IMPURITY_C
#             plot_interval=VISUAL_UPDATE_STEP,
#             output_prefix=prefix
#         )
#         run_end_time = time.time()

#         steps_to_save = [500, 1000]
#         for step in steps_to_save:
#             visualize_grain_structure(state, orientation_theta, atom_type, step, c_level,
#                           filename=f"outputs/microstructures/{prefix}/grain_structure_step_{step}_{prefix}.png")

#             events = get_event_rates(
#                 state, orientation_theta, orientation_phi, T, atom_type, introduce_defects(state, atom_type),
#                 LATTICE_SIZE, STATES['W'], STATES['Re'], STATES['C'], step=step, debug_step=1000, impurity_c=c_level
#             )
#             visualize_possible_events(events, step, c_level)

#         visualize_final_state(
#             state, atom_type, orientation_theta, impurity_c=c_level,
#             title=f"Final Lattice State (CET Study, C={c_level*100:.1f}%)",
#             filename=f"output_images/final_state_{prefix}.png"
#         )
#         np.save(f'output_images/final_state_{prefix}.npy', state)
#         np.save(f'output_images/final_atom_type_{prefix}.npy', atom_type)
#         print(f"Simulation complete for {prefix}. Total simulated time: {total_time:.4e} s")
#         print(f"Run wall-clock time: {run_end_time - run_start_time:.2f} s")
#         print(f"Metrics saved to 'outputs/metrics/{prefix}/metrics_{prefix}.csv'. Plots in 'outputs/metrics/{prefix}/' and 'outputs/microstructures/{prefix}/'.")

#         df = pd.read_csv(f'outputs/metrics/{prefix}/metrics_{prefix}.csv')
#         step_500 = df[df['Step'] == 500]
#         if not step_500.empty:
#             metrics_at_500['grain_sizes'].append(step_500['AvgGrainSize'].iloc[0])
#             metrics_at_500['defect_densities'].append(step_500['DefectDensity'].iloc[0])
#             metrics_at_500['aspect_ratios'].append(step_500['AspectRatio'].iloc[0])

#     plot_combined_metrics(carbon_levels, metrics_at_500['grain_sizes'], 
#                          metrics_at_500['defect_densities'], metrics_at_500['aspect_ratios'],
#                          save_dir="outputs/metrics")

#     end_time = time.time()
#     print(f"\nTotal wall-clock runtime for all runs: {end_time - start_time:.2f} s")

# if __name__ == "__main__":
#     main()

import numpy as np
import time
import os
import pandas as pd
from kmc_simulation import run_kmc
from lattice_init import initialize_lattice, save_lattice
from visualization import visualize_final_state, visualize_grain_structure
from defects import introduce_defects
from kmc_event_rates import get_event_rates
from constants import LATTICE_SIZE, N_STEPS, T_SUB, DEFECT_PROB, N_SEEDS, T_MELT, NU_DEP, STATES, IMPURITY_C, VISUAL_UPDATE_STEP, METRIC_UPDATE_STEP
from graphs import plot_metrics, plot_combined_metrics, save_microstructure_image, plot_grain_size_distribution
def main():
    """
    Main script to run KMC simulation for microstructure control in AM (CET study).
    Runs multiple simulations with varying carbon impurity levels to study CET.
    """
    print("Starting KMC simulation for microstructure control...")
    start_time = time.time()

    carbon_levels = [0.0, 0.1, 0.2]  # 0%, 10%, 20%
    metrics_at_500 = {'grain_sizes': [], 'defect_densities': [], 'aspect_ratios': []}

    for c_level in carbon_levels:
        print(f"\nRunning simulation with IMPURITY_C = {c_level*100:.1f}%")
        prefix = f"impurity_c_{int(c_level*100)}"
        os.makedirs(f"outputs/metrics/{prefix}", exist_ok=True)
        os.makedirs(f"outputs/microstructures/{prefix}", exist_ok=True)
        os.makedirs("output_images", exist_ok=True)

        state, orientation_theta, orientation_phi, T, atom_type = initialize_lattice(
            lattice_size=LATTICE_SIZE,
            n_seeds=N_SEEDS,
            T_sub=T_SUB,
            random_seed=42,
            impurity_c=c_level
        )

        save_lattice(state, orientation_theta, orientation_phi, T, atom_type, prefix=f"init_{prefix}")
        print(f"Initial lattice setup complete for {prefix}. Saved to 'init_{prefix}_*.npy'.")

        G = (T_MELT - T_SUB) / LATTICE_SIZE
        R = NU_DEP * 1e-16
        gr_ratio = G / R if R > 0 else float('inf')
        print(f"G/R ratio: {gr_ratio:.2e} K·s/unit (lower values promote equiaxed)")

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

        # Save final grain structure
        visualize_grain_structure(
            state, orientation_theta, atom_type, N_STEPS, c_level,
            filename=f"outputs/microstructures/{prefix}/final_grain_structure_{prefix}.png"
        )

        visualize_final_state(
            state, atom_type, orientation_theta, impurity_c=c_level,
            title=f"Final Lattice State (CET Study, C={c_level*100:.1f}%)",
            filename=f"output_images/final_state_{prefix}.png"
        )
        np.save(f'output_images/final_state_{prefix}.npy', state)
        np.save(f'output_images/final_atom_type_{prefix}.npy', atom_type)
        print(f"Simulation complete for {prefix}. Total simulated time: {total_time:.4e} s")
        print(f"Run wall-clock time: {run_end_time - run_start_time:.2f} s")
        print(f"Metrics saved to 'outputs/metrics/{prefix}/metrics_{prefix}.csv'. Plots in 'outputs/metrics/{prefix}/' and 'outputs/microstructures/{prefix}/'.")

        df = pd.read_csv(f'outputs/metrics/{prefix}/metrics_{prefix}.csv')
        step_500 = df[df['Step'] == METRIC_UPDATE_STEP]
        if not step_500.empty:
            metrics_at_500['grain_sizes'].append(step_500['AvgGrainSize'].iloc[0])
            metrics_at_500['defect_densities'].append(step_500['DefectDensity'].iloc[0])
            metrics_at_500['aspect_ratios'].append(step_500['AspectRatio'].iloc[0])

    plot_combined_metrics(carbon_levels, metrics_at_500['grain_sizes'], 
                         metrics_at_500['defect_densities'], metrics_at_500['aspect_ratios'],
                         save_dir="outputs/metrics")

    end_time = time.time()
    print(f"\nTotal wall-clock runtime for all runs: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    main()