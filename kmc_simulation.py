# import numpy as np
# import random
# import pandas as pd
# import os
# from kmc_event_rates import get_event_rates
# from defects import introduce_defects
# from utils import compute_metrics, get_clusters
# from constants import LATTICE_SIZE, N_STEPS, RATE_THRESHOLD, METRIC_UPDATE_STEP, VISUAL_UPDATE_STEP, STATES
# from lattice_init import initialize_lattice
# from visualization import plot_lattice


# def run_kmc(L, n_steps, temp, defect_fraction, n_seeds, impurity_c, plot_interval=500, output_prefix="default"):
#     """
#     Runs the KMC simulation for multi-material thin film growth, favoring columnar growth.
#     """
#     state, orientation_theta, orientation_phi, T, atom_type = initialize_lattice(
#         lattice_size=L, n_seeds=n_seeds, T_sub=temp, impurity_c=impurity_c
#     )

#     defects = introduce_defects(state, atom_type)
#     total_time = 0.0
#     metrics_data = []
#     grain_sizes_list = []
#     defect_densities_list = []
#     anisotropy_metrics_list = []
#     nucleation_count = 0

#     for step in range(n_steps):
#         events = get_event_rates(
#             state, orientation_theta, orientation_phi, T, atom_type, defects, L,
#             states_w=STATES['W'], states_re=STATES['Re'], states_c=STATES['C'],
#             step=step, debug_step=1000, impurity_c=impurity_c  # Fixed: Pass impurity_c
#         )

#         if not events:
#             print(f"Step {step}: No more events possible. Ending simulation.")
#             break

#         total_rate = sum(e[2] for e in events)
#         if total_rate <= 0:
#             print(f"Step {step}: Total rate zero. Ending simulation.")
#             break
#         if total_rate < RATE_THRESHOLD:
#             print(f"Step {step}: Total rate {total_rate:.2e} below threshold {RATE_THRESHOLD}. Ending simulation.")
#             break

#         r = random.random() * total_rate
#         cumulative = 0.0
#         chosen_event = None
#         for e in events:
#             cumulative += e[2]
#             if cumulative >= r:
#                 chosen_event = e
#                 break

#         if chosen_event is None:
#             print(f"Step {step}: No event selected. Skipping.")
#             continue

#         etype, pos, rate, target, atom = chosen_event
#         i, j, k = pos

#         if etype == 'dep':
#             state[i, j, k] = atom
#             atom_type[i, j, k] = atom
#             orientation_theta[i, j, k] = np.random.uniform(0, np.pi)
#             orientation_phi[i, j, k] = np.random.uniform(0, 2 * np.pi)
#         elif etype == 'det':
#             state[i, j, k] = STATES['Empty']
#             atom_type[i, j, k] = STATES['Empty']
#             orientation_theta[i, j, k] = 0.0
#             orientation_phi[i, j, k] = 0.0
#         elif etype == 'evap':
#             state[i, j, k] = STATES['Empty']
#             atom_type[i, j, k] = STATES['Empty']
#             orientation_theta[i, j, k] = 0.0
#             orientation_phi[i, j, k] = 0.0
#         elif etype == 'diff':
#             if target is not None:
#                 ti, tj, tk = target
#                 state[ti, tj, tk] = state[i, j, k]
#                 atom_type[ti, tj, tk] = atom_type[i, j, k]
#                 orientation_theta[ti, tj, tk] = orientation_theta[i, j, k]
#                 orientation_phi[ti, tj, tk] = orientation_phi[i, j, k]
#                 state[i, j, k] = STATES['Empty']
#                 atom_type[i, j, k] = STATES['Empty']
#                 orientation_theta[i, j, k] = 0.0
#                 orientation_phi[i, j, k] = 0.0
#         elif etype == 'nuc':
#             state[i, j, k] = atom
#             atom_type[i, j, k] = atom
#             orientation_theta[i, j, k] = np.random.uniform(0, np.pi)
#             orientation_phi[i, j, k] = np.random.uniform(0, 2 * np.pi)
#             nucleation_count += 1
#         elif etype == 'att':
#             if target is not None:
#                 ni, nj, nk = target
#                 state[i, j, k] = atom
#                 atom_type[i, j, k] = atom
#                 orientation_theta[i, j, k] = orientation_theta[ni, nj, nk]
#                 orientation_phi[i, j, k] = orientation_phi[ni, nj, nk]

#         dt = -np.log(random.random()) / total_rate
#         total_time += dt

#         if step % METRIC_UPDATE_STEP == 0 or step == n_steps - 1:  # Fixed: Use METRIC_UPDATE_STEP
#             clusters, _ = get_clusters(state, orientation_theta)
#             aspect_ratio, coverage, sizes, impurity_counts = compute_metrics(clusters, state, atom_type)
#             defect_density = np.sum(defects) / defects.size if defects.size > 0 else 0.0
#             avg_grain_size = np.mean(sizes) if sizes else 0.0
#             print(f"Step {step}: C_Count={impurity_counts[3]}, AvgGrainSize={avg_grain_size:.2f}, "
#                   f"AspectRatio={aspect_ratio:.2f}, DefectDensity={defect_density:.4f}, NucleationCount={nucleation_count}")
#             grain_sizes_list.append(avg_grain_size)
#             defect_densities_list.append(defect_density)
#             anisotropy_metrics_list.append(aspect_ratio)
#             metrics_data.append({
#                 'Step': step,
#                 'Time': total_time,
#                 'Coverage': coverage,
#                 'AspectRatio': aspect_ratio,
#                 'AvgGrainSize': avg_grain_size,
#                 'DefectDensity': defect_density,
#                 'W_Count': impurity_counts[1],
#                 'Re_Count': impurity_counts[2],
#                 'C_Count': impurity_counts[3],
#                 'NucleationCount': nucleation_count
#             })

#         if plot_interval and step % plot_interval == 0:
#             print(f"Step {step}/{n_steps} — Time: {total_time:.4e} s — Events: {len(events)}")
#             plot_lattice(state, step=step, impurity_c=impurity_c, 
#                          filename=f"outputs/microstructures/{output_prefix}/lattice_midZ_step_{step}_{output_prefix}.png")

#     if metrics_data:
#         os.makedirs(f'outputs/metrics/{output_prefix}', exist_ok=True)
#         df = pd.DataFrame(metrics_data)
#         df.to_csv(f'outputs/metrics/{output_prefix}/metrics_{output_prefix}.csv', index=False)

#     print(f"Simulation completed at step {step}, total time: {total_time:.2e} s")
#     return state, atom_type, total_time, orientation_theta, orientation_phi


import numpy as np
import random
import pandas as pd
import os
from kmc_event_rates import get_event_rates
from defects import introduce_defects
from utils import compute_metrics, get_clusters
from constants import LATTICE_SIZE, N_STEPS, RATE_THRESHOLD, METRIC_UPDATE_STEP, STATES
from lattice_init import initialize_lattice
from visualization import plot_lattice

def run_kmc(L, n_steps, temp, defect_fraction, n_seeds, impurity_c, output_prefix="default"):
    """
    Runs the KMC simulation for multi-material thin film growth, favoring columnar growth.
    """
    state, orientation_theta, orientation_phi, T, atom_type = initialize_lattice(
        lattice_size=L, n_seeds=n_seeds, T_sub=temp, impurity_c=impurity_c
    )

    defects = introduce_defects(state, atom_type)
    total_time = 0.0
    metrics_data = []
    grain_sizes_list = []
    defect_densities_list = []
    anisotropy_metrics_list = []
    nucleation_count = 0

    for step in range(n_steps):
        events = get_event_rates(
            state, orientation_theta, orientation_phi, T, atom_type, defects, L,
            states_w=STATES['W'], states_re=STATES['Re'], states_c=STATES['C'],
            step=step, debug_step=1000, impurity_c=impurity_c
        )

        if not events:
            print(f"Step {step}: No more events possible. Ending simulation.")
            break

        total_rate = sum(e[2] for e in events)
        if total_rate <= 0:
            print(f"Step {step}: Total rate zero. Ending simulation.")
            break
        if total_rate < RATE_THRESHOLD:
            print(f"Step {step}: Total rate {total_rate:.2e} below threshold {RATE_THRESHOLD}. Ending simulation.")
            break

        r = random.random() * total_rate
        cumulative = 0.0
        chosen_event = None
        for e in events:
            cumulative += e[2]
            if cumulative >= r:
                chosen_event = e
                break

        if chosen_event is None:
            print(f"Step {step}: No event selected. Skipping.")
            continue

        etype, pos, rate, target, atom = chosen_event
        i, j, k = pos

        if etype == 'dep':
            state[i, j, k] = atom
            atom_type[i, j, k] = atom
            orientation_theta[i, j, k] = np.random.uniform(0, np.pi)
            orientation_phi[i, j, k] = np.random.uniform(0, 2 * np.pi)
        elif etype == 'det':
            state[i, j, k] = STATES['Empty']
            atom_type[i, j, k] = STATES['Empty']
            orientation_theta[i, j, k] = 0.0
            orientation_phi[i, j, k] = 0.0
        elif etype == 'evap':
            state[i, j, k] = STATES['Empty']
            atom_type[i, j, k] = STATES['Empty']
            orientation_theta[i, j, k] = 0.0
            orientation_phi[i, j, k] = 0.0
        elif etype == 'diff':
            if target is not None:
                ti, tj, tk = target
                state[ti, tj, tk] = state[i, j, k]
                atom_type[ti, tj, tk] = atom_type[i, j, k]
                orientation_theta[ti, tj, tk] = orientation_theta[i, j, k]
                orientation_phi[ti, tj, tk] = orientation_phi[i, j, k]
                state[i, j, k] = STATES['Empty']
                atom_type[i, j, k] = STATES['Empty']
                orientation_theta[i, j, k] = 0.0
                orientation_phi[i, j, k] = 0.0
        elif etype == 'nuc':
            state[i, j, k] = atom
            atom_type[i, j, k] = atom
            orientation_theta[i, j, k] = np.random.uniform(0, np.pi)
            orientation_phi[i, j, k] = np.random.uniform(0, 2 * np.pi)
            nucleation_count += 1
        elif etype == 'att':
            if target is not None:
                ni, nj, nk = target
                state[i, j, k] = atom
                atom_type[i, j, k] = atom
                orientation_theta[i, j, k] = orientation_theta[ni, nj, nk]
                orientation_phi[i, j, k] = orientation_phi[ni, nj, nk]

        dt = -np.log(random.random()) / total_rate
        total_time += dt

        if step % METRIC_UPDATE_STEP == 0 or step == n_steps - 1:
            clusters, _ = get_clusters(state, orientation_theta)
            aspect_ratio, coverage, sizes, impurity_counts = compute_metrics(clusters, state, atom_type)
            defect_density = np.sum(defects) / defects.size if defects.size > 0 else 0.0
            avg_grain_size = np.mean(sizes) if sizes else 0.0
            print(f"Step {step}: C_Count={impurity_counts[3]}, AvgGrainSize={avg_grain_size:.2f}, "
                  f"AspectRatio={aspect_ratio:.2f}, DefectDensity={defect_density:.4f}, NucleationCount={nucleation_count}")
            grain_sizes_list.append(avg_grain_size)
            defect_densities_list.append(defect_density)
            anisotropy_metrics_list.append(aspect_ratio)
            metrics_data.append({
                'Step': step,
                'Time': total_time,
                'Coverage': coverage,
                'AspectRatio': aspect_ratio,
                'AvgGrainSize': avg_grain_size,
                'DefectDensity': defect_density,
                'W_Count': impurity_counts[1],
                'Re_Count': impurity_counts[2],
                'C_Count': impurity_counts[3],
                'NucleationCount': nucleation_count
            })

        if step == n_steps - 1:
            print(f"Final Step {step}/{n_steps} — Time: {total_time:.4e} s — Events: {len(events)}")
            plot_lattice(state, step=step, impurity_c=impurity_c, 
                         filename=f"outputs/microstructures/{output_prefix}/final_lattice_midZ_{output_prefix}.png")

    if metrics_data:
        os.makedirs(f'outputs/metrics/{output_prefix}', exist_ok=True)
        df = pd.DataFrame(metrics_data)
        df.to_csv(f'outputs/metrics/{output_prefix}/metrics_{output_prefix}.csv', index=False)

    print(f"Simulation completed at step {step}, total time: {total_time:.2e} s")
    return state, atom_type, total_time, orientation_theta, orientation_phi