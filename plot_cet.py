# analyze_results.py
"""
Post-processing script for CET-KMC simulation.
Automatically collects metrics from outputs/impurity_c_*/ folders
and generates scientific figures.

Author: Govinda + ChatGPT
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ========================
# CONFIG
# ========================
input_root = "outputs"
outdir = "analysis_plots"
os.makedirs(outdir, exist_ok=True)

# ========================
# Discover metric files
# ========================
files = {}
for path in sorted(glob.glob(os.path.join(input_root, "impurity_c_*", "metrics_*.csv"))):
    # Example path: outputs/impurity_c_10/metrics_10.csv
    folder = os.path.basename(os.path.dirname(path))   # impurity_c_10
    c_level = folder.split("_")[-1]                   # "10"
    label = f"{c_level}% C"
    files[label] = path

print("📂 Found files:", files)

# ========================
# Load data
# ========================
data = {}
for label, f in files.items():
    df = pd.read_csv(f)
    data[label] = df

# ========================
# Line Plots (evolution)
# ========================
def plot_evolution(metric, ylabel, filename, logy=False):
    plt.figure(figsize=(7,5))
    for label, df in data.items():
        plt.plot(df["Step"], df[metric], label=label)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()

plot_evolution("AspectRatio", "Aspect Ratio", "aspect_ratio_vs_step.png")
plot_evolution("DefectDensity", "Defect Density [a.u.]", "defect_density_vs_step.png", logy=True)
plot_evolution("EquiaxedFraction", "Equiaxed Fraction", "equiaxed_fraction_vs_step.png")

# ========================
# Bar Plots (final values)
# ========================
final_metrics = pd.DataFrame({
    label: {
        "Final AspectRatio": df["AspectRatio"].iloc[-1],
        "Final DefectDensity": df["DefectDensity"].iloc[-1],
        "Final EquiaxedFraction": df["EquiaxedFraction"].iloc[-1],
    }
    for label, df in data.items()
}).T

def plot_bar(metric, ylabel, filename):
    plt.figure(figsize=(6,4))
    final_metrics[metric].plot(kind="bar", color=["steelblue","darkorange","seagreen"])
    plt.ylabel(ylabel)
    plt.xlabel("Carbon Impurity Level")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()

plot_bar("Final AspectRatio", "Aspect Ratio", "bar_aspect_ratio.png")
plot_bar("Final DefectDensity", "Defect Density [a.u.]", "bar_defect_density.png")
plot_bar("Final EquiaxedFraction", "Equiaxed Fraction", "bar_eqfrac.png")

# ========================
# Combined Summary Plot
# ========================
plt.figure(figsize=(12,8))

# 1. Line: Aspect Ratio
plt.subplot(2,2,1)
for label, df in data.items():
    plt.plot(df["Step"], df["AspectRatio"], label=label)
plt.title("Aspect Ratio Evolution")
plt.xlabel("Step"); plt.ylabel("Aspect Ratio"); plt.legend(); plt.grid(alpha=0.3)

# 2. Line: Defect Density
plt.subplot(2,2,2)
for label, df in data.items():
    plt.plot(df["Step"], df["DefectDensity"], label=label)
plt.title("Defect Density Evolution")
plt.xlabel("Step"); plt.ylabel("Defect Density"); plt.yscale("log"); plt.grid(alpha=0.3)

# 3. Line: Equiaxed Fraction
plt.subplot(2,2,3)
for label, df in data.items():
    plt.plot(df["Step"], df["EquiaxedFraction"], label=label)
plt.title("Equiaxed Fraction Evolution")
plt.xlabel("Step"); plt.ylabel("Eq. Fraction"); plt.grid(alpha=0.3)

# 4. Bar: Final Defects
plt.subplot(2,2,4)
final_metrics["Final DefectDensity"].plot(kind="bar", color=["steelblue","darkorange","seagreen"])
plt.title("Final Defect Density")
plt.ylabel("Defect Density"); plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "summary_combined.png"))
plt.close()

print(f"✅ Analysis complete. Plots saved in {outdir}/")
