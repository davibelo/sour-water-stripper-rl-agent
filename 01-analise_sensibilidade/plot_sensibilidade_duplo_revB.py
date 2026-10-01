from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import re
import unicodedata
import numpy as np


DATA_PATH = Path(__file__).resolve().parent / "AnaliseSensibilidade_revC.xlsx"
SHEET_NAME = "Dados"
OUTPUT_DIR = Path(__file__).resolve().parent / "figuras"
GENERATE_COLOR = True
GENERATE_MONO = True
CREATE_SUBFOLDERS = True
COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_str.lower()).strip("_")
    return slug or "figura"


def plot_metric(df, x_col, column_template, y_label, title, filename, mono=False, cases=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df[x_col]
    cases = list(cases) if cases is not None else list(range(1, 11))
    for idx, case_idx in enumerate(cases):
        col = column_template.format(case_idx)
        if mono:
            marker = MARKERS[idx % len(MARKERS)]
            ax.plot(
                x,
                df[col],
                color="black",
                linewidth=1,
                marker=marker,
                markersize=3,
                label=f"Caso {case_idx}",
            )
        else:
            color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            ax.plot(x, df[col], linewidth=1, color=color, label=f"Caso {case_idx}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4, which='both')
    ax.legend()

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    if "%" in y_label:
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(2))
    elif "topo" in y_label.lower():
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
    elif "fundo" in y_label.lower():
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    fig.tight_layout()
    filepath = Path(filename)
    if not filepath.is_absolute():
        filepath = OUTPUT_DIR / filepath
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)


def plot_dual_metric(df, x_col, h2s_template, nh3_template, title, filename, mono=False, cases=None):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = df[x_col]
    cases = list(cases) if cases is not None else list(range(1, 11))

    ax2 = ax1.twinx()

    for idx, case_idx in enumerate(cases):
        h2s_col = h2s_template.format(case_idx)
        nh3_col = nh3_template.format(case_idx)

        if mono:
            marker = MARKERS[idx % len(MARKERS)]
            ax1.plot(
                x,
                df[h2s_col],
                color="black",
                linewidth=1,
                marker=marker,
                markersize=3,
                linestyle="-",
                label=f"H2S Caso {case_idx}",
            )
            ax2.plot(
                x,
                df[nh3_col],
                color="gray",
                linewidth=1,
                marker=marker,
                markersize=3,
                linestyle="--",
                label=f"NH3 Caso {case_idx}",
            )
        else:
            color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            ax1.plot(x, df[h2s_col], linewidth=1.5, color=color, linestyle="-", label=f"H2S Caso {case_idx}")
            ax2.plot(x, df[nh3_col], linewidth=1.5, color=color, linestyle="--", label=f"NH3 Caso {case_idx}")

    ax1.set_xlabel(x_col)
    ax1.set_ylabel("Recuperação de H2S [%]", color="tab:blue")
    ax2.set_ylabel("Perda de NH3 [%]", color="tab:red")
    ax1.set_title(title)
    ax1.grid(True, linestyle="--", alpha=0.4, which='both')
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(2))
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_minor_locator(MultipleLocator(2))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    filepath = Path(filename)
    if not filepath.is_absolute():
        filepath = OUTPUT_DIR / filepath
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    x_col = "Carga térmica refervedor [Gcal/h]"

    # Calculate T fundo - T topo for each case
    for case_idx in range(1, 11):
        fundo_col = f"T fundo [°C] - Caso {case_idx}"
        topo_col = f"T topo [°C] - Caso {case_idx}"
        delta_col = f"T fundo - T topo [°C] - Caso {case_idx}"
        if fundo_col in df.columns and topo_col in df.columns:
            df[delta_col] = df[fundo_col] - df[topo_col]

    metrics = [
        {
            "column_template": "T fundo [°C] - Caso {}",
            "y_label": "Temperatura de fundo [°C]",
            "title": "Carga térmica refervedor x T Fundo",
            "base_filename": "q_ref_vs_t_fundo",
        },
        {
            "column_template": "T topo [°C] - Caso {}",
            "y_label": "Temperatura de topo [°C]",
            "title": "Carga térmica  refervedor x T Topo",
            "base_filename": "q_ref_vs_t_topo",
        },
        {
            "column_template": "T fundo - T topo [°C] - Caso {}",
            "y_label": "T fundo - T topo [°C]",
            "title": "Carga térmica refervedor x (T fundo - T topo)",
            "base_filename": "q_ref_vs_t_fundo_t_topo",
        },
        {
            "column_template": "Recuperação H2S [%] - Caso {}",
            "y_label": "Recuperação de H2S [%]",
            "title": "Carga térmica refervedor x Recuperação H2S",
            "base_filename": "q_ref_vs_recuperacao_h2s",
        },
        {
            "column_template": "Perda NH3 [%] - Caso {}",
            "y_label": "Perda de NH3 [%]",
            "title": "Carga térmica refervedor x Perda NH3",
            "base_filename": "q_ref_vs_perda_nh3",
        },
    ]
    case_groups = [
        {"cases": [5, 6, 7, 8, 9, 10], "label": "influência contaminantes"},
        {"cases": [1, 8, 2], "label": "influência vazão carga"},
        {"cases": [3, 8, 4], "label": "influência temperatura carga"},
        {"cases": [5, 6, 8, 10], "label": "influência razão constante"},
    ]

    for metric in metrics:
        base = metric["base_filename"]
        if GENERATE_COLOR:
            plot_metric(
                df,
                x_col,
                metric["column_template"],
                metric["y_label"],
                metric["title"],
                f"color/todos os casos/{base}.png",
                mono=False,
                cases=range(1, 11),
            )
        if GENERATE_MONO:
            plot_metric(
                df,
                x_col,
                metric["column_template"],
                metric["y_label"],
                metric["title"],
                f"mono/todos os casos/{base}_mono.png",
                mono=True,
                cases=range(1, 11),
            )
        for group in case_groups:
            suffix = slugify(group["label"])
            title = f"{metric['title']} - {group['label']}"
            if CREATE_SUBFOLDERS:
                color_subdir = f"color/{suffix}"
                mono_subdir = f"mono/{suffix}"
            else:
                color_subdir = "color"
                mono_subdir = "mono"

            if GENERATE_COLOR:
                filename_color = f"{color_subdir}/{base}_{suffix}.png"
                plot_metric(
                    df,
                    x_col,
                    metric["column_template"],
                    metric["y_label"],
                    title,
                    filename_color,
                    mono=False,
                    cases=group["cases"],
                )
            if GENERATE_MONO:
                filename_mono = f"{mono_subdir}/{base}_{suffix}_mono.png"
                plot_metric(
                    df,
                    x_col,
                    metric["column_template"],
                    metric["y_label"],
                    title,
                    filename_mono,
                    mono=True,
                    cases=group["cases"],
                )

    h2s_template = "Recuperação H2S [%] - Caso {}"
    nh3_template = "Perda NH3 [%] - Caso {}"

    if GENERATE_COLOR:
        plot_dual_metric(
            df,
            x_col,
            h2s_template,
            nh3_template,
            "Carga térmica refervedor x Recuperação H2S e Perda NH3",
            "color/todos os casos/q_ref_vs_h2s_nh3_dual.png",
            mono=False,
            cases=range(1, 11),
        )
    if GENERATE_MONO:
        plot_dual_metric(
            df,
            x_col,
            h2s_template,
            nh3_template,
            "Carga térmica refervedor x Recuperação H2S e Perda NH3",
            "mono/todos os casos/q_ref_vs_h2s_nh3_dual_mono.png",
            mono=True,
            cases=range(1, 11),
        )

    for group in case_groups:
        suffix = slugify(group["label"])
        title = f"Carga térmica refervedor x Recuperação H2S e Perda NH3 - {group['label']}"
        if CREATE_SUBFOLDERS:
            color_subdir = f"color/{suffix}"
            mono_subdir = f"mono/{suffix}"
        else:
            color_subdir = "color"
            mono_subdir = "mono"

        if GENERATE_COLOR:
            filename_color = f"{color_subdir}/q_ref_vs_h2s_nh3_dual_{suffix}.png"
            plot_dual_metric(
                df,
                x_col,
                h2s_template,
                nh3_template,
                title,
                filename_color,
                mono=False,
                cases=group["cases"],
            )
        if GENERATE_MONO:
            filename_mono = f"{mono_subdir}/q_ref_vs_h2s_nh3_dual_{suffix}_mono.png"
            plot_dual_metric(
                df,
                x_col,
                h2s_template,
                nh3_template,
                title,
                filename_mono,
                mono=True,
                cases=group["cases"],
            )


if __name__ == "__main__":
    main()
