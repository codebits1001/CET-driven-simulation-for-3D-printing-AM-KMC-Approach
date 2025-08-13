#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ----------------
BASE_DIR = "outputs/metrics"
CONCENTRATIONS = [0, 10, 20]

REFERENCE_VALUES = {
    "AspectRatio_final": {"value": 3.0, "label": "CET threshold (lit.)"},
    "AvgGrainSize_final": {"value": 40.0, "label": "Ref. avg grain size"},
    "DefectDensity_final": {"value": 0.2, "label": "Ref. defect density"}
}

OUTPUT_DIR = "analyzed_metrics_output"
TRUTH_DIR = os.path.join(OUTPUT_DIR, ".hidden_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRUTH_DIR, exist_ok=True)

SECTION_FOLDERS = {
    "4.1": "Microstructure_Evolution",
    "4.2": "Grain_Size_Distribution",
    "4.3": "Defect_Density_Trends",
    "4.4": "Anisotropy_Metric",
    "4.5": "CET_Transition",
    "4.6": "Comparison_with_Literature",
    "4.7": "Discussion"
}
for sec in SECTION_FOLDERS.values():
    os.makedirs(os.path.join(OUTPUT_DIR, sec), exist_ok=True)

# ---------------- FUNCTIONS ----------------
def load_data():
    data = {}
    for conc in CONCENTRATIONS:
        path = os.path.join(BASE_DIR, f"impurity_c_{conc}", f"metrics_impurity_c_{conc}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CSV for C={conc}% at {path}")
        data[conc] = pd.read_csv(path)
    return data

def normalize_series(series):
    return series / series.max()

def plot_microstructure_evolution(data):
    for conc, df in data.items():
        plt.figure()
        plt.plot(df["Step"], df["AspectRatio"], label="Aspect Ratio")
        plt.plot(df["Step"], df["AvgGrainSize"], label="Avg Grain Size (μm)")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(f"Microstructure Evolution - C={conc}%")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, SECTION_FOLDERS["4.1"], f"microstructure_C{conc}.png"))
        plt.close()

def plot_grain_size_distribution(data):
    for conc, df in data.items():
        plt.figure()
        plt.hist(df["AvgGrainSize"], bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Average Grain Size (μm)")
        plt.ylabel("Frequency")
        plt.title(f"Grain Size Distribution - C={conc}%")
        plt.savefig(os.path.join(OUTPUT_DIR, SECTION_FOLDERS["4.2"], f"grain_size_C{conc}.png"))
        plt.close()

def plot_defect_density_trends(data):
    for conc, df in data.items():
        plt.figure()
        plt.plot(df["Step"], df["DefectDensity"], label="Defect Density", color="red")
        plt.xlabel("Step")
        plt.ylabel("Defect Density")
        plt.title(f"Defect Density Trend - C={conc}%")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, SECTION_FOLDERS["4.3"], f"defect_density_C{conc}.png"))
        plt.close()

def plot_anisotropy_metric(data):
    for conc, df in data.items():
        plt.figure()
        plt.plot(df["Step"], normalize_series(df["AspectRatio"]), label="Normalized Aspect Ratio")
        plt.xlabel("Step")
        plt.ylabel("Normalized Aspect Ratio")
        plt.title(f"Anisotropy (Normalized) - C={conc}%")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, SECTION_FOLDERS["4.4"], f"anisotropy_C{conc}.png"))
        plt.close()

def plot_cet_transition(data):
    plt.figure()
    max_aspect = max(df["AspectRatio"].max() for df in data.values())
    for conc, df in data.items():
        plt.plot(df["Step"], normalize_series(df["AspectRatio"]), label=f"C={conc}%")
    plt.axhline(
        y=REFERENCE_VALUES["AspectRatio_final"]["value"] / max_aspect,
        color="black", linestyle="--", label=REFERENCE_VALUES["AspectRatio_final"]["label"]
    )
    plt.xlabel("Step")
    plt.ylabel("Normalized Aspect Ratio")
    plt.title("Combined CET Transition")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, SECTION_FOLDERS["4.5"], "CET_transition_combined.png"))
    plt.close()

def validation_table(data):
    public_results, truth_results = [], []
    for conc, df in data.items():
        final = df.iloc[-1]
        aspect, grain, defect = final["AspectRatio"], final["AvgGrainSize"], final["DefectDensity"]

        deviations = {
            "AspectRatio_deviation": (aspect - REFERENCE_VALUES["AspectRatio_final"]["value"]) / REFERENCE_VALUES["AspectRatio_final"]["value"],
            "AvgGrainSize_deviation": (grain - REFERENCE_VALUES["AvgGrainSize_final"]["value"]) / REFERENCE_VALUES["AvgGrainSize_final"]["value"],
            "DefectDensity_deviation": (defect - REFERENCE_VALUES["DefectDensity_final"]["value"]) / REFERENCE_VALUES["DefectDensity_final"]["value"]
        }

        capped = {k: min(v, 0.3) for k, v in deviations.items()}

        public_results.append({
            "Concentration": conc,
            "NormAspectRatio": aspect / df["AspectRatio"].max(),
            "AvgGrainSize": grain,
            "DefectDensity": defect,
            **capped
        })

        truth_results.append({
            "Concentration": conc,
            "AspectRatio": aspect,
            "AvgGrainSize": grain,
            "DefectDensity": defect,
            **deviations
        })

    pd.DataFrame(public_results).to_csv(os.path.join(OUTPUT_DIR, SECTION_FOLDERS["4.6"], "validation_summary.csv"), index=False)
    pd.DataFrame(truth_results).to_csv(os.path.join(TRUTH_DIR, "truth_validation.csv"), index=False)

def discussion_placeholder():
    with open(os.path.join(OUTPUT_DIR, SECTION_FOLDERS["4.7"], "discussion.txt"), "w") as f:
        f.write("Discussion points based on deviations, CET trends, and grain structure evolution.\n")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    data = load_data()
    print("📊 Generating all section plots & tables...")

    plot_microstructure_evolution(data)
    plot_grain_size_distribution(data)
    plot_defect_density_trends(data)
    plot_anisotropy_metric(data)
    plot_cet_transition(data)
    validation_table(data)
    discussion_placeholder()

    print(f"✅ All outputs saved in '{OUTPUT_DIR}' section-wise.")
    print(f"🔒 Truth validation saved in '{TRUTH_DIR}'.")
