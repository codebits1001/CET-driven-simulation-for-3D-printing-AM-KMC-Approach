# import os
# import random
# import numpy as np
# import pandas as pd

# from thermal_solver import update_temperature_cet as update_temperature
# from kmc_event_rates import get_event_rates
# from defects import introduce_defects
# from lattice_init import initialize_lattice
# from metrics import compute_metrics, compute_CET, detect_CET_transition
# from constants import (
#     LATTICE_SIZE, N_STEPS, RATE_THRESHOLD, METRIC_UPDATE_STEP,
#     CET_CHECK_INTERVAL, T_SUB, T_MELT, NU_DEP, VOXEL_SIZE, RANDOM_SEED, ATOMIC_SPACING_W, DEFECT_ID
# )

# def run_kmc(
#     L: int = LATTICE_SIZE,
#     n_steps: int = N_STEPS,
#     temp: float = T_SUB,
#     defect_fraction: float = 0.0,
#     n_seeds: int = 5,
#     impurity_c: float = 0.0,
#     output_prefix: str = "cet_run",
# ):
#     np.random.seed(RANDOM_SEED)
#     random.seed(RANDOM_SEED)

#     output_dir = f"outputs/{output_prefix}"
#     os.makedirs(output_dir, exist_ok=True)

#     state, theta, phi, T, atom_type = initialize_lattice(
#         lattice_size=L, n_seeds=n_seeds, T_sub=temp, impurity_c=impurity_c
#     )

#     defects_mask, defect_density = introduce_defects(
#         state, atom_type, T, apply_to_state=False
#     )

#     G = (T_MELT - T_SUB) / (L * VOXEL_SIZE)  # [K/m]
#     R = NU_DEP * 2.74e-10 / VOXEL_SIZE  # [1/s] normalized
#     # NEW: physical growth speed and ratio in SI units
#     R_phys = NU_DEP * ATOMIC_SPACING_W           # [m/s]
#     G_over_R_phys = G / R_phys                   # [K·s/m²]

#     total_time = 0.0
#     metrics_data = []
#     nucleation_count = 0
#     cet_detected = False

#     for step in range(n_steps):
#         if step % 20 == 0:
#             T = np.nan_to_num(T, nan=T_SUB)  # Prevent NaN in T
#             T = update_temperature(T, state, dt=1e-6)

#         events = get_event_rates(
#             state, theta, phi, T, atom_type, defects_mask, L,
#             1, 2, 3,  # Hardcoded STATES['W'], ['Re'], ['C']
#             step=step, debug_step=1000, impurity_c=impurity_c
#         )

#         total_rate = sum(e[2] for e in events) if events else 0.0
#         if (not events) or (total_rate < 1e-25) or (not np.isfinite(total_rate)):
#             print(f"Terminating at step {step}: no valid events (rate={total_rate:.2e})")
#             break

#         r = random.random() * total_rate
#         cumulative = 0.0
#         chosen = None
#         for e in events:
#             cumulative += e[2]
#             if cumulative >= r:
#                 chosen = e
#                 break

#         if chosen is None:
#             chosen = events[-1]

#         etype, pos, rate, target, atom = chosen
#         i, j, k = pos

#         if etype == b'dep':
#             state[i, j, k] = atom
#             atom_type[i, j, k] = atom
#             theta[i, j, k] = np.random.uniform(0, np.pi)
#             phi[i, j, k] = np.random.uniform(0, 2 * np.pi)

#         elif etype in (b'det', b'evap'):
#             state[i, j, k] = 0  # Empty
#             atom_type[i, j, k] = 0
#             theta[i, j, k] = 0.0
#             phi[i, j, k] = 0.0

#         elif etype == b'diff' and target[0] != -1:
#             ti, tj, tk = target
#             state[ti, tj, tk] = state[i, j, k]
#             atom_type[ti, tj, tk] = atom_type[i, j, k]
#             theta[ti, tj, tk] = theta[i, j, k]
#             phi[ti, tj, tk] = phi[i, j, k]
#             state[i, j, k] = 0
#             atom_type[i, j, k] = 0
#             theta[i, j, k] = 0.0
#             phi[i, j, k] = 0.0

#         elif etype == b'nuc':
#             state[i, j, k] = atom
#             atom_type[i, j, k] = atom
#             theta[i, j, k] = np.random.uniform(0, np.pi)
#             phi[i, j, k] = np.random.uniform(0, 2 * np.pi)
#             nucleation_count += 1

#         elif etype == b'att' and target[0] != -1:
#             ni, nj, nk = target
#             state[i, j, k] = atom
#             atom_type[i, j, k] = atom
#             theta[i, j, k] = theta[ni, nj, nk]
#             phi[i, j, k] = phi[ni, nj, nk]

#         dt = max(-np.log(max(1e-12, random.random())) / total_rate, 1e-12)
#         total_time += dt

#         if step % METRIC_UPDATE_STEP == 0:
#             defects_mask, defect_density = introduce_defects(
#                 state, atom_type, T, apply_to_state=False
#             )

#         if (step % METRIC_UPDATE_STEP == 0) or (step == n_steps - 1):
#             m = compute_metrics(state, theta, phi, defects=defects_mask, voxel_size=VOXEL_SIZE)
#             defect_voxels = np.sum(atom_type == DEFECT_ID)
#             defect_frac = defect_voxels / atom_type.size

#             m["Defect_voxel_count"] = int(defect_voxels)
#             m["DefectDensity"] = float(defect_frac)
#     # ----

#             if (not cet_detected) and detect_CET_transition(m):
#                 cet_detected = True
#                 print(f"CET detected at step {step} (G/R={G / R:.2e})")

#             w_count = int((state == 1).sum())
#             re_count = int((state == 2).sum())
#             c_count = int((state == 3).sum())

#             row = {
#                 "Step": step,
#                 "Time": total_time,
#                 "AspectRatio": m["AspectRatio"],
#                 "EquiaxedFraction": m["EquiaxedFraction"],
#                 "NucleationDensity": m["NucleationDensity"],
#                 "DefectDensity": m["DefectDensity"],
#                 "AvgGrainSize": m["AvgGrainSize"],
#                 "GrainCount": m["GrainCount"],
#                 "W_Count": w_count,
#                 "Re_Count": re_count,
#                 "C_Count": c_count,
#                 "NucleationCount": nucleation_count,
#                 "G_over_R": (G / R) if R > 0 else np.inf,
#                 # >>> NEW SI-consistent fields <<<
#                 "G_phys": G,                    # [K/m]
#                 "R_phys": R_phys,               # [m/s]
#                 "G_over_R_phys": G_over_R_phys, # [K·s/m²]
#                 "CET_Class": compute_CET(state, theta, phi, voxel_size=VOXEL_SIZE),
#                 "CET_Detected": cet_detected,

#             }
#             metrics_data.append(row)

#             print(
#                 f"Step {step}: AR={row['AspectRatio']:.2f}, "
#                 f"EqFrac={row['EquiaxedFraction']:.2f}, "
#                 f"NucDens={row['NucleationDensity']:.3e}, "
#                 f"DefectDens={row['DefectDensity']:.3e}, "
#                 f"CET={row['CET_Class']}, "
#                 f"Detected={row['CET_Detected']}, "
#                 f"Time={row['Time']:.2e}s"
#             )

#     if metrics_data:
#         df = pd.DataFrame(metrics_data)
#         output_path = os.path.join(output_dir, "metrics.csv")
#         df.to_csv(output_path, index=False)
#         print(f"Metrics saved to {output_path}")

#     print(f"Completed {step + 1} steps in {total_time:.2e} s")
#     return state, atom_type, total_time, theta, phi


import os
import random
import numpy as np
import pandas as pd

from thermal_solver import update_temperature_cet as update_temperature
from kmc_event_rates import get_event_rates
from defects import introduce_defects
from lattice_init import initialize_lattice
from metrics import compute_metrics, compute_CET, detect_CET_transition
from constants import (
    LATTICE_SIZE, N_STEPS, RATE_THRESHOLD, METRIC_UPDATE_STEP,
    CET_CHECK_INTERVAL, T_SUB, T_MELT, NU_DEP, VOXEL_SIZE, RANDOM_SEED,
    ATOMIC_SPACING_W, DEFECT_ID
)

def run_kmc(
    L: int = LATTICE_SIZE,
    n_steps: int = N_STEPS,
    temp: float = T_SUB,
    defect_fraction: float = 0.0,   # <-- used as per-event defect insertion probability
    n_seeds: int = 5,
    impurity_c: float = 0.0,
    output_prefix: str = "cet_run",
):
    """
    KMC microstructure evolution with natural defect injection.

    defect_fraction : float in [0, 1]
        Probability that the just-updated voxel becomes a defect (DEFECT_ID).
        Keeps original flow but lets defects accumulate naturally.
    """
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    output_dir = f"outputs/{output_prefix}"
    os.makedirs(output_dir, exist_ok=True)

    # --- Init lattice ---
    state, theta, phi, T, atom_type = initialize_lattice(
        lattice_size=L, n_seeds=n_seeds, T_sub=temp, impurity_c=impurity_c
    )

    # Initial (diagnostic) defect field from your existing helper; not applied to state
    defects_mask, defect_density = introduce_defects(
        state, atom_type, T, apply_to_state=False
    )

    # Thermal/kinetic fields and SI-consistent ratios
    G = (T_MELT - T_SUB) / (L * VOXEL_SIZE)  # [K/m]
    R = NU_DEP * 2.74e-10 / VOXEL_SIZE       # normalized [1/s] (kept from your code)
    R_phys = NU_DEP * ATOMIC_SPACING_W       # [m/s]
    G_over_R_phys = G / R_phys               # [K·s/m²]

    total_time = 0.0
    metrics_data = []
    nucleation_count = 0
    cet_detected = False

    for step in range(n_steps):
        # occasional temperature update
        if step % 20 == 0:
            T = np.nan_to_num(T, nan=T_SUB)
            T = update_temperature(T, state, dt=1e-6)

        # assemble events
        events = get_event_rates(
            state, theta, phi, T, atom_type, defects_mask, L,
            1, 2, 3,  # STATES['W'], STATES['Re'], STATES['C'] hardcoded as in your code
            step=step, debug_step=1000, impurity_c=impurity_c
        )

        total_rate = sum(e[2] for e in events) if events else 0.0
        if (not events) or (total_rate < 1e-25) or (not np.isfinite(total_rate)):
            print(f"Terminating at step {step}: no valid events (rate={total_rate:.2e})")
            break

        # pick event (Gillespie)
        r = random.random() * total_rate
        cumulative = 0.0
        chosen = None
        for e in events:
            cumulative += e[2]
            if cumulative >= r:
                chosen = e
                break
        if chosen is None:
            chosen = events[-1]

        etype, pos, rate, target, atom = chosen
        i, j, k = pos

        # ----- apply event to lattice -----
        if etype == b'dep':
            state[i, j, k] = atom
            atom_type[i, j, k] = atom
            theta[i, j, k] = np.random.uniform(0, np.pi)
            phi[i, j, k] = np.random.uniform(0, 2 * np.pi)

        elif etype in (b'det', b'evap'):
            state[i, j, k] = 0
            atom_type[i, j, k] = 0
            theta[i, j, k] = 0.0
            phi[i, j, k] = 0.0

        elif etype == b'diff' and target[0] != -1:
            ti, tj, tk = target
            state[ti, tj, tk] = state[i, j, k]
            atom_type[ti, tj, tk] = atom_type[i, j, k]
            theta[ti, tj, tk] = theta[i, j, k]
            phi[ti, tj, tk] = phi[i, j, k]
            state[i, j, k] = 0
            atom_type[i, j, k] = 0
            theta[i, j, k] = 0.0
            phi[i, j, k] = 0.0
            # for diff, the "updated" site is the target voxel
            i, j, k = ti, tj, tk  # so the patch below applies to the filled site

        elif etype == b'nuc':
            state[i, j, k] = atom
            atom_type[i, j, k] = atom
            theta[i, j, k] = np.random.uniform(0, np.pi)
            phi[i, j, k] = np.random.uniform(0, 2 * np.pi)
            nucleation_count += 1

        elif etype == b'att' and target[0] != -1:
            ni, nj, nk = target
            state[i, j, k] = atom
            atom_type[i, j, k] = atom
            theta[i, j, k] = theta[ni, nj, nk]
            phi[i, j, k] = phi[ni, nj, nk]
        # ----------------------------------

        # ====== PATCH: natural defect injection after a site update ======
        # Interpret defect_fraction as a small per-update probability.
        # This keeps your flow intact while allowing defects to accumulate.
        if defect_fraction > 0.0 and random.random() < defect_fraction:
            atom_type[i, j, k] = DEFECT_ID
            state[i, j, k] = DEFECT_ID
            theta[i, j, k] = 0.0
            phi[i, j, k] = 0.0
        # =================================================================

        # time advance
        dt = max(-np.log(max(1e-12, random.random())) / total_rate, 1e-12)
        total_time += dt

        # refresh diagnostic defect mask for metrics at your cadence
        if step % METRIC_UPDATE_STEP == 0:
            defects_mask, defect_density = introduce_defects(
                state, atom_type, T, apply_to_state=False
            )

        # metrics logging
        if (step % METRIC_UPDATE_STEP == 0) or (step == n_steps - 1):
            m = compute_metrics(state, theta, phi, defects=defects_mask, voxel_size=VOXEL_SIZE)

            # defect density from actual lattice labeling
            defect_voxels = int(np.sum(atom_type == DEFECT_ID))
            defect_frac = float(defect_voxels / atom_type.size)

            m["Defect_voxel_count"] = defect_voxels
            m["DefectDensity"] = defect_frac

            if (not cet_detected) and detect_CET_transition(m):
                cet_detected = True
                print(f"CET detected at step {step} (G/R={G / R:.2e})")

            w_count = int((state == 1).sum())
            re_count = int((state == 2).sum())
            c_count = int((state == 3).sum())

            row = {
                "Step": step,
                "Time": total_time,
                "AspectRatio": m["AspectRatio"],
                "EquiaxedFraction": m["EquiaxedFraction"],
                "NucleationDensity": m["NucleationDensity"],
                "DefectDensity": m["DefectDensity"],
                "AvgGrainSize": m["AvgGrainSize"],
                "GrainCount": m["GrainCount"],
                "W_Count": w_count,
                "Re_Count": re_count,
                "C_Count": c_count,
                "NucleationCount": nucleation_count,
                "G_over_R": (G / R) if R > 0 else np.inf,
                "G_phys": G,
                "R_phys": R_phys,
                "G_over_R_phys": G_over_R_phys,
                "CET_Class": compute_CET(state, theta, phi, voxel_size=VOXEL_SIZE),
                "CET_Detected": cet_detected,
            }
            metrics_data.append(row)

            print(
                f"Step {step}: AR={row['AspectRatio']:.2f}, "
                f"EqFrac={row['EquiaxedFraction']:.2f}, "
                f"NucDens={row['NucleationDensity']:.3e}, "
                f"DefectDens={row['DefectDensity']:.3e}, "
                f"CET={row['CET_Class']}, "
                f"Detected={row['CET_Detected']}, "
                f"Time={row['Time']:.2e}s"
            )

    if metrics_data:
        df = pd.DataFrame(metrics_data)
        output_path = os.path.join(output_dir, "metrics.csv")
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")

    print(f"Completed {step + 1} steps in {total_time:.2e} s")
    return state, atom_type, total_time, theta, phi
