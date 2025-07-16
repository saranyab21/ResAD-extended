import os
import pandas as pd
import matplotlib.pyplot as plt

# Set plot style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

def plot_classwise_failure(csv_path, metric="Pixel AUC", top_k_fail=3, title=None, save_path=None):
    df = pd.read_csv(csv_path)
    if "Class Name" not in df.columns or metric not in df.columns:
        raise ValueError(f"Required columns not found in {csv_path}")
    
    # Sort by metric (descending = better performance)
    df_sorted = df.sort_values(by=metric, ascending=False).reset_index(drop=True)

    # Highlight the worst-performing classes in red
    colors = ["lightcoral" if i >= len(df_sorted) - top_k_fail else "#1f77b4" for i in range(len(df_sorted))]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(df_sorted["Class Name"], df_sorted[metric], color=colors)
    plt.gca().invert_yaxis()  # Show highest AUROC on top

    # Add AUROC value labels
    for bar, score in zip(bars, df_sorted[metric]):
        plt.text(bar.get_width() - 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", ha='right', va='center', fontsize=10,
                 color='white' if bar.get_width() > 0.9 else 'black')

    plt.xlabel(f"{metric} (4-shot)")
    plt.title(title or f"Class-wise {metric}")
    plt.xlim(0.75, 1.00)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.figtext(0.99, 0.01, "*Highlighted bars = 3 lowest-performing classes",
                    ha='right', fontsize=9, style='italic')
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Dataset 1: MVTec
    plot_classwise_failure(
        csv_path="./checkpoints/mvtec_to_visa_4shot/mvtec_to_visa_4shot_epoch19_metrics.csv",
        metric="Pixel AUC",
        top_k_fail=3,
        title="Class-wise Pixel-level AUROC on MVTecAD\n(ResAD, 4-shot setting)",
        save_path="mvtec_classwise_auroc.png"
    )

    # Dataset 2: VisA
    plot_classwise_failure(
        csv_path="./checkpoints/mvtec_to_visa_4shot/mvtec_to_visa_4shot_epoch19_metrics.csv",
        metric="Pixel AUC",
        top_k_fail=3,
        title="Class-wise Pixel-level AUROC on VisA\n(ResAD, 4-shot setting)",
        save_path="visa_classwise_auroc.png"
    )
