from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_ID = "04"


def min_max_scale(values, mins, maxs):
    return (values - mins) / (maxs - mins)


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / f"sim_results-{DATA_ID}.csv"
    images_dir = base_dir / "imagens"
    outputs_dir = base_dir / "arquivos"

    images_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    inputs = ["Qfeed", "Tfeed", "cH2S", "cNH3", "Qreb", "Pcolumn", "Ecolumn"]
    outputs = ["Tbottom", "Ttop"]

    mins = np.array([47500, 118.95, 875, 750, 2.0, 5.7, 66.5], dtype=float)
    maxs = np.array([52500, 125.05, 3500, 3000, 6.0, 6.3, 73.5], dtype=float)

    out_mins = df[outputs].min()
    out_maxs = df[outputs].max()

    scaled_df = df.copy()
    scaled_df[inputs] = min_max_scale(df[inputs].to_numpy(dtype=float), mins, maxs)
    scaled_df[outputs] = min_max_scale(df[outputs].to_numpy(dtype=float), out_mins.to_numpy(), out_maxs.to_numpy())

    descriptive_stats = df[inputs + outputs].describe()
    descriptive_stats.to_csv(outputs_dir / f"descriptive_stats-{DATA_ID}.csv")

    correlation = df[inputs + outputs].corr()
    correlation.to_csv(outputs_dir / f"correlation-{DATA_ID}.csv")

    output_minmax = pd.DataFrame({"min": out_mins, "max": out_maxs})
    output_minmax.to_csv(outputs_dir / f"output_minmax-{DATA_ID}.csv")

    scaled_df.to_csv(outputs_dir / f"sim_results_scaled-{DATA_ID}.csv", index=False)

    plt.figure(figsize=(12, 8))
    for i, col in enumerate(inputs, start=1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(images_dir / f"hist_inputs-{DATA_ID}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    for i, col in enumerate(outputs, start=1):
        plt.subplot(1, 2, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(images_dir / f"hist_outputs-{DATA_ID}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    sns.boxplot(data=scaled_df[inputs], orient="h")
    plt.tight_layout()
    plt.savefig(images_dir / f"boxplot_inputs_scaled-{DATA_ID}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=scaled_df[outputs], orient="h")
    plt.tight_layout()
    plt.savefig(images_dir / f"boxplot_outputs_scaled-{DATA_ID}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, cmap="coolwarm", annot=True, fmt=".2f")
    plt.tight_layout()
    plt.savefig(images_dir / f"correlation_heatmap-{DATA_ID}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="Tbottom", y="Ttop", s=20, alpha=0.7)
    plt.tight_layout()
    plt.savefig(images_dir / f"scatter_outputs-{DATA_ID}.png", dpi=200)
    plt.close()

    # Pair Plot (if data size is manageable)
    sns.pairplot(df.sample(n=min(500, len(df))), diag_kind='kde')
    plt.suptitle("Feature Relationships (Pair Plot) in Scaled Data", y=1.02)
    plt.savefig(images_dir / f"pair_plot-{DATA_ID}.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(len(outputs), len(inputs), figsize=(20, 8), sharey="row")
    for i, out_col in enumerate(outputs):
        for j, in_col in enumerate(inputs):
            ax = axes[i, j] if len(outputs) > 1 else axes[j]
            sns.scatterplot(x=df[in_col], y=df[out_col], s=10, alpha=0.6, ax=ax)
            if i == len(outputs) - 1:
                ax.set_xlabel(in_col)
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(out_col)
            else:
                ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(images_dir / f"scatter_inputs_outputs-{DATA_ID}.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
