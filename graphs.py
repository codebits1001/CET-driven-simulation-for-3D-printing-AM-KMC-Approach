# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from constants import EPSILON

# def plot_metrics(step, grain_sizes, defect_densities, anisotropy_metrics, save_dir="outputs/metrics"):
#     """
#     Plot and save microstructure metrics:
#     - Average grain size
#     - Defect density
#     - Anisotropy metric
    
#     Args:
#         step (int): Current simulation step.
#         grain_sizes (list): Average grain sizes over time.
#         defect_densities (list): Defect density over time.
#         anisotropy_metrics (list): Anisotropy values over time.
#         save_dir (str): Directory where plots will be saved.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     steps = np.arange(0, step + 1, max(1, step // max(1, len(grain_sizes) - 1)))

#     # Grain size evolution
#     plt.figure()
#     plt.plot(steps, grain_sizes, marker='o')
#     plt.xlabel("Step")
#     plt.ylabel("Average Grain Size")
#     plt.title("Grain Size Evolution")
#     plt.grid(True)
#     plt.savefig(os.path.join(save_dir, f"grain_size_step_{step}.png"))
#     plt.close()

#     # Defect density evolution
#     plt.figure()
#     plt.plot(steps, defect_densities, marker='o', color='red')
#     plt.xlabel("Step")
#     plt.ylabel("Defect Density")
#     plt.title("Defect Density Evolution")
#     plt.grid(True)
#     plt.savefig(os.path.join(save_dir, f"defect_density_step_{step}.png"))
#     plt.close()

#     # Anisotropy metric evolution
#     plt.figure()
#     plt.plot(steps, anisotropy_metrics, marker='o', color='green')
#     plt.xlabel("Step")
#     plt.ylabel("Anisotropy Metric")
#     plt.title("Anisotropy Metric Evolution")
#     plt.grid(True)
#     plt.savefig(os.path.join(save_dir, f"anisotropy_step_{step}.png"))
#     plt.close()

# def plot_combined_metrics(carbon_levels, grain_sizes, defect_densities, aspect_ratios, save_dir="outputs/metrics"):
#     """
#     Plot combined metrics (grain size, defect density, aspect ratio) across carbon levels at a given step.
    
#     Args:
#         carbon_levels (list): List of carbon impurity levels (e.g., [0.0, 0.1, 0.2]).
#         grain_sizes (list): Average grain sizes for each carbon level.
#         defect_densities (list): Defect densities for each carbon level.
#         aspect_ratios (list): Aspect ratios for each carbon level.
#         save_dir (str): Directory to save the combined plot.
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

#     labels = [f"{c*100:.0f}% C" for c in carbon_levels]
    
#     # Grain size
#     ax1.bar(labels, grain_sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
#     ax1.set_title("Average Grain Size")
#     ax1.set_ylabel("Grain Size (voxels)")
    
#     # Defect density
#     ax2.bar(labels, defect_densities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
#     ax2.set_title("Defect Density")
#     ax2.set_ylabel("Defect Density")
    
#     # Aspect ratio
#     ax3.bar(labels, aspect_ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
#     ax3.set_title("Aspect Ratio")
#     ax3.set_ylabel("Aspect Ratio")
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "combined_metrics_step_500.png"))
#     plt.close()

# def save_microstructure_image(step, lattice, save_dir="outputs/microstructures"):
#     """
#     Save a 2D slice of the 3D lattice as an image for visual inspection.

#     Args:
#         step (int): Current simulation step.
#         lattice (np.ndarray): 3D lattice state array.
#         save_dir (str): Directory to save the slice image.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Take middle slice along z-axis
#     mid_slice = lattice[:, :, lattice.shape[2] // 2]

#     plt.figure(figsize=(6, 6))
#     plt.imshow(mid_slice.T, origin='lower', cmap='viridis', interpolation='nearest')
#     plt.colorbar(label="Atom Type / State")
#     plt.title(f"Microstructure Slice at Step {step}")
#     plt.savefig(os.path.join(save_dir, f"microstructure_step_{step}.png"))
#     plt.close()
    
# def plot_grain_size_distribution(step, grain_sizes, save_dir="outputs/metrics"):
#     os.makedirs(save_dir, exist_ok=True)
#     plt.figure()
#     plt.hist(grain_sizes, bins=20, color='blue', alpha=0.7)
#     plt.xlabel("Grain Size (voxels)")
#     plt.ylabel("Frequency")
#     plt.title(f"Grain Size Distribution at Step {step}")
#     plt.grid(True)
#     plt.savefig(os.path.join(save_dir, f"grain_size_dist_step_{step}.png"))
#     plt.close()



import os
import numpy as np
import matplotlib.pyplot as plt
from constants import EPSILON, METRIC_UPDATE_STEP  # Added METRIC_UPDATE_STEP here


def plot_metrics(step, grain_sizes, defect_densities, anisotropy_metrics, save_dir="outputs/metrics"):
    """
    Plot and save microstructure metrics:
    - Average grain size
    - Defect density
    - Anisotropy metric
    """
    os.makedirs(save_dir, exist_ok=True)

    steps = np.arange(0, step + 1, max(1, step // max(1, len(grain_sizes) - 1)))

    # Grain size evolution
    plt.figure()
    plt.plot(steps, grain_sizes, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Average Grain Size")
    plt.title("Grain Size Evolution")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"grain_size_step_{step}.png"))
    plt.close()

    # Defect density evolution
    plt.figure()
    plt.plot(steps, defect_densities, marker='o', color='red')
    plt.xlabel("Step")
    plt.ylabel("Defect Density")
    plt.title("Defect Density Evolution")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"defect_density_step_{step}.png"))
    plt.close()

    # Anisotropy metric evolution
    plt.figure()
    plt.plot(steps, anisotropy_metrics, marker='o', color='green')
    plt.xlabel("Step")
    plt.ylabel("Anisotropy Metric")
    plt.title("Anisotropy Metric Evolution")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"anisotropy_step_{step}.png"))
    plt.close()


def plot_combined_metrics(carbon_levels, grain_sizes, defect_densities, aspect_ratios, save_dir="outputs/metrics"):
    """
    Plot combined metrics (grain size, defect density, aspect ratio) across carbon levels.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prevent shape mismatch errors
    if not (len(carbon_levels) == len(grain_sizes) == len(defect_densities) == len(aspect_ratios)):
        print("[Warning] Skipping combined metrics plot — data length mismatch.")
        print(f"Lengths: carbon_levels={len(carbon_levels)}, "
              f"grain_sizes={len(grain_sizes)}, "
              f"defect_densities={len(defect_densities)}, "
              f"aspect_ratios={len(aspect_ratios)}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    labels = [f"{c*100:.0f}% C" for c in carbon_levels]
    
    # Grain size
    ax1.bar(labels, grain_sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title("Average Grain Size")
    ax1.set_ylabel("Grain Size (voxels)")
    
    # Defect density
    ax2.bar(labels, defect_densities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title("Defect Density")
    ax2.set_ylabel("Defect Density")
    
    # Aspect ratio
    ax3.bar(labels, aspect_ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_title("Aspect Ratio")
    ax3.set_ylabel("Aspect Ratio")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"combined_metrics_step_{METRIC_UPDATE_STEP}.png"))
    plt.close()


def save_microstructure_image(step, lattice, save_dir="outputs/microstructures"):
    """
    Save a 2D slice of the 3D lattice as an image for visual inspection.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Take middle slice along z-axis
    mid_slice = lattice[:, :, lattice.shape[2] // 2]

    plt.figure(figsize=(6, 6))
    plt.imshow(mid_slice.T, origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Atom Type / State")
    plt.title(f"Microstructure Slice at Step {step}")
    plt.savefig(os.path.join(save_dir, f"microstructure_step_{step}.png"))
    plt.close()


def plot_grain_size_distribution(step, grain_sizes, save_dir="outputs/metrics"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.hist(grain_sizes, bins=20, color='blue', alpha=0.7)
    plt.xlabel("Grain Size (voxels)")
    plt.ylabel("Frequency")
    plt.title(f"Grain Size Distribution at Step {step}")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"grain_size_dist_step_{step}.png"))
    plt.close()
