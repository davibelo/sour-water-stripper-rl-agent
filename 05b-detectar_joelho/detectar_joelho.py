import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ID = "04"
VALIDACAO_QREB_MIN = 2.4
VALIDACAO_QREB_MAX = 3.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(os.path.dirname(BASE_DIR), "05-surrogate", f"sim_results-{DATA_ID}.csv")
FIG_DIR = BASE_DIR
LOG_PATH = os.path.join(BASE_DIR, "detectar_joelho.log")
os.makedirs(FIG_DIR, exist_ok=True)

lines = []
w = lines.append

df = pd.read_csv(CSV_PATH)
df["delta_T"] = df["Tbottom"] - df["Ttop"]

w("=" * 70)
w("DETECCAO DO JOELHO OPERACIONAL — Tbottom vs Qreb")
w("=" * 70)
w(f"\nTotal de amostras: {len(df)}")
w(f"Faixa de Qreb: [{df['Qreb'].min():.4f}, {df['Qreb'].max():.4f}]")
w(f"Faixa de Tbottom: [{df['Tbottom'].min():.2f}, {df['Tbottom'].max():.2f}]")
w(f"Faixa de delta_T: [{df['delta_T'].min():.2f}, {df['delta_T'].max():.2f}]")

N_BINS = 36
df["Qreb_bin"] = pd.cut(df["Qreb"], bins=N_BINS)
binned = df.groupby("Qreb_bin", observed=True).agg(
    Qreb_mid=("Qreb", "median"),
    Tbottom_med=("Tbottom", "median"),
    Tbottom_mean=("Tbottom", "mean"),
    Tbottom_q25=("Tbottom", lambda x: x.quantile(0.25)),
    Tbottom_q75=("Tbottom", lambda x: x.quantile(0.75)),
    Ttop_med=("Ttop", "median"),
    delta_T_med=("delta_T", "median"),
    delta_T_mean=("delta_T", "mean"),
    delta_T_q25=("delta_T", lambda x: x.quantile(0.25)),
    delta_T_q75=("delta_T", lambda x: x.quantile(0.75)),
    n=("Qreb", "count"),
).dropna().sort_values("Qreb_mid").reset_index(drop=True)

w(f"\nBins com dados: {len(binned)} (de {N_BINS} bins)")
w(f"\n{'Qreb_mid':>10s} | {'Tb_med':>8s} | {'dT_med':>8s} | {'Tt_med':>8s} | {'n':>4s}")
w("-" * 50)
for _, row in binned.iterrows():
    w(f"{row['Qreb_mid']:10.3f} | {row['Tbottom_med']:8.2f} | "
      f"{row['delta_T_med']:8.2f} | {row['Ttop_med']:8.2f} | {int(row['n']):4d}")

x = binned["Qreb_mid"].values
y_tb = binned["Tbottom_med"].values
y_dt = binned["delta_T_med"].values

w("\n" + "=" * 70)
w("METODO 1: Kneedle — Tbottom vs Qreb")
w("=" * 70)

x_norm = (x - x.min()) / (x.max() - x.min())
y_tb_norm = (y_tb - y_tb.min()) / (y_tb.max() - y_tb.min())
diff_kneedle = y_tb_norm - x_norm
knee_idx_kn = np.argmax(diff_kneedle)
qreb_knee_kn = x[knee_idx_kn]
tb_knee_kn = y_tb[knee_idx_kn]

w(f"\nKneedle: maximo desvio da diagonal em Qreb = {qreb_knee_kn:.3f}")
w(f"  Tbottom nesse ponto: {tb_knee_kn:.2f} C")
w(f"  Desvio normalizado: {diff_kneedle[knee_idx_kn]:.4f}")

w("\n" + "=" * 70)
w("METODO 2: Maxima curvatura — Tbottom vs Qreb")
w("=" * 70)

try:
    from scipy.signal import savgol_filter
    wlen = min(7, len(x) if len(x) % 2 == 1 else len(x) - 1)
    y_tb_smooth = savgol_filter(y_tb, window_length=wlen, polyorder=3)
    w(f"\nSuavizacao Savitzky-Golay: window={wlen}, polyorder=3")
except ImportError:
    y_tb_smooth = y_tb
    w("\nScipy nao disponivel, usando dados brutos")

dy = np.gradient(y_tb_smooth, x)
d2y = np.gradient(dy, x)
curvature = np.abs(d2y) / (1 + dy**2) ** 1.5

knee_idx_cv = np.argmax(curvature)
qreb_knee_cv = x[knee_idx_cv]
tb_knee_cv = y_tb[knee_idx_cv]

w(f"\nMaxima curvatura em Qreb = {qreb_knee_cv:.3f}")
w(f"  Tbottom nesse ponto: {tb_knee_cv:.2f} C")
w(f"  Curvatura: {curvature[knee_idx_cv]:.6f}")

w(f"\n  Derivada primeira (dTb/dQreb) por bin:")
for i in range(len(x)):
    w(f"    Qreb={x[i]:.3f}: dTb/dQreb={dy[i]:+.3f}, d2Tb/dQreb2={d2y[i]:+.3f}")

w("\n" + "=" * 70)
w("METODO 3: Pico de delta_T = Tbottom - Ttop")
w("=" * 70)

peak_idx = np.argmax(y_dt)
qreb_peak_dt = x[peak_idx]
dt_peak = y_dt[peak_idx]

w(f"\nPico de delta_T (mediana por bin) em Qreb = {qreb_peak_dt:.3f}")
w(f"  delta_T nesse ponto: {dt_peak:.2f} C")

above_90pct = y_dt > 0.9 * dt_peak
if np.any(above_90pct):
    qreb_90_lo = x[above_90pct].min()
    qreb_90_hi = x[above_90pct].max()
    w(f"  Regiao com delta_T > 90% do pico: Qreb in [{qreb_90_lo:.3f}, {qreb_90_hi:.3f}]")

w("\n" + "=" * 70)
w("METODO 4: Transicao dTbottom/dQreb — ponto de desaceleracao")
w("=" * 70)

dy_abs = np.abs(dy)
if len(dy_abs) > 3:
    half = len(dy_abs) // 2
    high_deriv_region = dy_abs[:half]
    if np.max(high_deriv_region) > 0:
        threshold = 0.3 * np.max(high_deriv_region)
        transition_idxs = np.where((dy_abs[:-1] > threshold) & (dy_abs[1:] < threshold))[0]
        if len(transition_idxs) > 0:
            tidx = transition_idxs[0]
            qreb_trans = (x[tidx] + x[tidx + 1]) / 2.0
            w(f"\nTransicao de dTb/dQreb (queda > 70%): Qreb ~ {qreb_trans:.3f}")
            w(f"  dTb/dQreb antes: {dy[tidx]:.3f}")
            w(f"  dTb/dQreb depois: {dy[tidx + 1]:.3f}")

w("\n" + "=" * 70)
w("RESUMO E VALIDACAO")
w("=" * 70)

qreb_knees = {
    "Kneedle": qreb_knee_kn,
    "Max curvatura": qreb_knee_cv,
    "Pico delta_T": qreb_peak_dt,
}
w(f"\nResultados dos metodos:")
for name, val in qreb_knees.items():
    w(f"  {name:20s}: Qreb = {val:.3f}")

all_knees = list(qreb_knees.values())
qreb_knee_mean = np.mean(all_knees)
qreb_knee_std = np.std(all_knees)
w(f"\nMedia dos metodos: Qreb = {qreb_knee_mean:.3f} +/- {qreb_knee_std:.3f}")

in_expected = [VALIDACAO_QREB_MIN <= v <= VALIDACAO_QREB_MAX for v in all_knees]
w(f"\nValidacao visual (esperado: Qreb entre {VALIDACAO_QREB_MIN} e {VALIDACAO_QREB_MAX}):")
for name, val in qreb_knees.items():
    ok = "OK" if VALIDACAO_QREB_MIN <= val <= VALIDACAO_QREB_MAX else "FORA"
    w(f"  {name:20s}: Qreb = {val:.3f} -> {ok}")

knee_in_range = sum(in_expected)
w(f"\n{knee_in_range}/{len(all_knees)} metodos confirmam joelho entre {VALIDACAO_QREB_MIN} e {VALIDACAO_QREB_MAX}")

knees_valid = [v for v, ok in zip(all_knees, in_expected) if ok]
if knees_valid:
    qreb_knee_best = np.median(knees_valid)
    w(f"\nJoelho adotado (mediana dos metodos dentro da faixa de validacao): Qreb = {qreb_knee_best:.3f}")
else:
    qreb_knee_best = np.median(all_knees)
    w(f"\nNenhum metodo dentro da faixa de validacao; mediana geral usada: Qreb = {qreb_knee_best:.3f}")
tb_at_knee_idx = np.argmin(np.abs(x - qreb_knee_best))
tb_at_knee = y_tb[tb_at_knee_idx]
dt_at_knee = y_dt[tb_at_knee_idx]

w(f"  Tbottom no joelho: {tb_at_knee:.2f} C")
w(f"  delta_T no joelho: {dt_at_knee:.2f} C")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Deteccao do Joelho Operacional — Stripper H2S/NH3", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.scatter(df["Qreb"], df["Tbottom"], s=3, alpha=0.3, color="gray", label="amostras")
ax.plot(x, y_tb, "b-o", markersize=5, linewidth=2, label="mediana por bin")
for name, val in qreb_knees.items():
    idx = np.argmin(np.abs(x - val))
    ax.axvline(val, linestyle="--", alpha=0.6, label=f"{name}: {val:.2f}")
ax.set_xlabel("Qreb (Gcal/h)")
ax.set_ylabel("Tbottom (C)")
ax.set_title("Tbottom vs Qreb")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(df["Qreb"], df["delta_T"], s=3, alpha=0.3, color="gray", label="amostras")
ax.plot(x, y_dt, "r-o", markersize=5, linewidth=2, label="mediana por bin")
ax.axvline(qreb_peak_dt, color="green", linestyle="--", linewidth=2, label=f"pico dT: Qreb={qreb_peak_dt:.2f}")
ax.set_xlabel("Qreb (Gcal/h)")
ax.set_ylabel("delta_T = Tbottom - Ttop (C)")
ax.set_title("delta_T vs Qreb")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.scatter(df["Qreb"], df["Ttop"], s=3, alpha=0.3, color="gray", label="amostras")
ttop_med = binned["Ttop_med"].values
ax.plot(x, ttop_med, "m-o", markersize=5, linewidth=2, label="mediana por bin")
ax.set_xlabel("Qreb (Gcal/h)")
ax.set_ylabel("Ttop (C)")
ax.set_title("Ttop vs Qreb")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(x, diff_kneedle, "g-o", markersize=5, linewidth=2)
ax.axvline(qreb_knee_kn, color="red", linestyle="--", linewidth=2,
           label=f"Kneedle: Qreb={qreb_knee_kn:.2f}")
ax.set_xlabel("Qreb (Gcal/h)")
ax.set_ylabel("Desvio da diagonal (norm)")
ax.set_title("Kneedle — Tbottom vs Qreb")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(x, curvature, "k-o", markersize=5, linewidth=2)
ax.axvline(qreb_knee_cv, color="red", linestyle="--", linewidth=2,
           label=f"Max curv: Qreb={qreb_knee_cv:.2f}")
ax.set_xlabel("Qreb (Gcal/h)")
ax.set_ylabel("Curvatura |f''|/(1+f'^2)^1.5")
ax.set_title("Curvatura — Tbottom vs Qreb")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.plot(x, dy, "b-o", markersize=5, linewidth=2, label="dTb/dQreb")
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.axvline(qreb_knee_best, color="red", linestyle="--", linewidth=2,
           label=f"Joelho: Qreb={qreb_knee_best:.2f}")
ax.set_xlabel("Qreb (Gcal/h)")
ax.set_ylabel("dTbottom / dQreb")
ax.set_title("Derivada de Tbottom vs Qreb")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "detectar_joelho.png")
plt.savefig(fig_path, dpi=150)
plt.close()

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nFigura salva em: {fig_path}")
print(f"Log salvo em: {LOG_PATH}")
