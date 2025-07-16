import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

def load_auroc(folder_base, prefix="mvtec_to_visa", shots=[2, 4, 6, 8]):
    results = {"mvtec_img": [], "mvtec_pix": [], "visa_img": [], "visa_pix": []}
    for shot in shots:
        folder = os.path.join(folder_base, f"{prefix}_{shot}shot")
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found.")
            continue
        # Get latest metrics CSV
        files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        if not files:
            continue
        latest_csv = sorted(files)[-1]
        df = pd.read_csv(os.path.join(folder, latest_csv))

        mvtec_df = df[df["Class Name"].isin([
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
            "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
        ])]
        visa_df = df[~df["Class Name"].isin(mvtec_df["Class Name"])]

        results["mvtec_img"].append(mvtec_df["Image AUC"].mean())
        results["mvtec_pix"].append(mvtec_df["Pixel AUC"].mean())
        results["visa_img"].append(visa_df["Image AUC"].mean())
        results["visa_pix"].append(visa_df["Pixel AUC"].mean())
    return results

def plot_shot_sensitivity(results, shots, save_path="shots_vs_auroc.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(shots, results["mvtec_img"], 'o-', color='royalblue', label="MVTec - Image AUROC")
    plt.plot(shots, results["mvtec_pix"], 's--', color='royalblue', label="MVTec - Pixel AUROC")
    plt.plot(shots, results["visa_img"], 'o-', color='forestgreen', label="VisA - Image AUROC")
    plt.plot(shots, results["visa_pix"], 's--', color='forestgreen', label="VisA - Pixel AUROC")

    plt.xlabel("Number of Reference Shots")
    plt.ylabel("AUROC Score")
    plt.title("Shot Sensitivity Analysis: AUROC vs Reference Shots")
    plt.xticks(shots)
    plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    plt.ylim(0.75, 1.00)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    checkpoint_base = "./checkpoints"
    shots = [2, 4, 6, 8]
    results = load_auroc(checkpoint_base, prefix="mvtec_to_visa", shots=shots)
    plot_shot_sensitivity(results, shots)
