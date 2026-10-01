"""
03_plot_histograms.py
Gera histogramas de H2S_Recover, NH3_Loss e n_iter_final para cada run.
Salva as figuras em final_results/figures/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("09-agentRL-DDPG-aspen-test/final_results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

RUNS = ["01"]
REWARD = "A"

COLS = ["H2S_Recover", "NH3_Loss", "n_iter_final"]
LABELS = {
    "H2S_Recover":  "H₂S Recovery",
    "NH3_Loss":     "NH₃ Loss",
    "n_iter_final": "Iterações totais",
}
UNITS = {
    "H2S_Recover":  "(frac.)",
    "NH3_Loss":     "(frac.)",
    "n_iter_final": "",
}
BINS = {
    "H2S_Recover":  30,
    "NH3_Loss":     30,
    "n_iter_final": None,  # será calculado automaticamente (inteiros)
}

for run in RUNS:
    path = RESULTS_DIR / f"sim_results_{run}_{REWARD}.csv"
    if not path.exists():
        print(f"[WARN] Arquivo não encontrado: {path}")
        continue

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Run {run} — Histogramas de métricas de simulação (Aspen)", fontsize=13)

    for ax, col in zip(axes, COLS):
        data = df[col].dropna()

        if col == "n_iter_final":
            bins = range(int(data.min()), int(data.max()) + 2)
            ax.hist(data, bins=bins, color="steelblue", edgecolor="white", align="left")
        else:
            ax.hist(data, bins=BINS[col], color="steelblue", edgecolor="white")

        ax.set_title(LABELS[col], fontsize=11)
        ax.set_xlabel(f"{LABELS[col]} {UNITS[col]}".strip(), fontsize=9)
        ax.set_ylabel("Contagem", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / f"hist_run{run}_{REWARD}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out}")

print("Concluído.")
