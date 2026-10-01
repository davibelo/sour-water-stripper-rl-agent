import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(os.path.dirname(BASE_DIR), "05-surrogate", "sim_results.csv")
LOG_PATH = os.path.join(BASE_DIR, "calibrar_reward_c.log")

df = pd.read_csv(CSV_PATH)
df["delta_T"] = df["Tbottom"] - df["Ttop"]

lines = []
w = lines.append

w("=" * 60)
w("CALIBRACAO DE PARAMETROS DA RECOMPENSA C")
w("=" * 60)

w(f"\nTotal de amostras: {len(df)}")
w(f"\n--- Estatisticas de delta_T = Tbottom - Ttop ---")
w(df["delta_T"].describe().to_string())

w(f"\n--- Percentis de delta_T ---")
for p in [5, 10, 25, 50, 75, 90, 95]:
    w(f"  P{p:02d}: {df['delta_T'].quantile(p/100):.4f}")

w(f"\n--- Estatisticas de Qreb ---")
w(df["Qreb"].describe().to_string())

qreb_min = df["Qreb"].min()
qreb_max = df["Qreb"].max()
qreb_nom = (qreb_min + qreb_max) / 2.0
w(f"\nQREB_NOM (centro da faixa): {qreb_nom:.4f}")

knee_mask = (df["Qreb"] >= 3.0) & (df["Qreb"] <= 4.5)
df_knee = df[knee_mask]
w(f"\n--- Regiao do joelho (Qreb entre 3.0 e 4.5) ---")
w(f"  Amostras nessa faixa: {len(df_knee)}")
if len(df_knee) > 0:
    w(f"  delta_T medio:   {df_knee['delta_T'].mean():.4f}")
    w(f"  delta_T mediana: {df_knee['delta_T'].median():.4f}")
    w(f"  delta_T std:     {df_knee['delta_T'].std():.4f}")

w("\n" + "=" * 60)
w("SUGESTAO DE PARAMETROS")
w("=" * 60)

delta_t_ref = df["delta_T"].median()
w(f"\ndelta_t_ref (mediana global de delta_T): {delta_t_ref:.4f}")

p75 = df["delta_T"].quantile(0.75)
p25 = df["delta_T"].quantile(0.25)
iqr = p75 - p25
sigma_iqr = iqr / 2.0
w(f"\nIQR de delta_T: {iqr:.4f}")
w(f"sigma sugerido (IQR / 2): {sigma_iqr:.4f}")

delta_t_range = df["delta_T"].max() - df["delta_T"].min()
sigma_range = delta_t_range / 6.0
w(f"\nRange de delta_T: {delta_t_range:.4f}")
w(f"sigma alternativo (range / 6): {sigma_range:.4f}")

w(f"\n--- Verificacao da tanh com os valores sugeridos ---")
w(f"  Usando delta_t_ref={delta_t_ref:.2f}, sigma={sigma_iqr:.2f}:")
test_points = [
    df["delta_T"].min(),
    p25,
    delta_t_ref,
    p75,
    df["delta_T"].max(),
]
labels = ["min", "P25", "mediana", "P75", "max"]
for lbl, val in zip(labels, test_points):
    tanh_val = np.tanh((val - delta_t_ref) / sigma_iqr)
    w(f"    delta_T={val:8.2f} ({lbl:>7s}) -> tanh = {tanh_val:+.4f}")

w(f"\n--- Verificacao com sigma alternativo (range/6) ---")
w(f"  Usando delta_t_ref={delta_t_ref:.2f}, sigma={sigma_range:.2f}:")
for lbl, val in zip(labels, test_points):
    tanh_val = np.tanh((val - delta_t_ref) / sigma_range)
    w(f"    delta_T={val:8.2f} ({lbl:>7s}) -> tanh = {tanh_val:+.4f}")

w(f"\n{'=' * 60}")
w("VALORES RECOMENDADOS PARA config.py")
w(f"{'=' * 60}")
w(f'  "delta_t_ref": {delta_t_ref:.1f},')
w(f'  "sigma": {sigma_iqr:.1f},')
w(f"{'=' * 60}")

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")