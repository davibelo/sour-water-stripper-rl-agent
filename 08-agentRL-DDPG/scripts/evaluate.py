import os
import sys
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from config import (
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    N_EVAL_EPISODES,
    QREB_INIT_HIGH,
    QREB_INIT_LOW,
    QREB_INIT_MID,
    QREB_MAX,
    QREB_MIN,
    SEED,
)
from surrogate_env import StripperSurrogateEnv

RUN = "01"
N_EPISODES = N_EVAL_EPISODES


def load_model(run_dir, reward_type):
    best_path = os.path.join(run_dir, "best_model.zip")
    final_path = os.path.join(run_dir, "final_model.zip")
    model_path = best_path if os.path.exists(best_path) else final_path

    vec_norm_path = os.path.join(run_dir, "vec_normalize.pkl")

    env = DummyVecEnv([lambda: Monitor(StripperSurrogateEnv(reward_type=reward_type, seed=SEED + 9999))])
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    model = DDPG.load(model_path, env=env)
    return model, env


def _get_base_env(env):
    try:
        return env.venv.envs[0].env
    except AttributeError:
        try:
            return env.envs[0].env
        except AttributeError:
            return None


def run_episodes(model, env, n_episodes, deterministic=True):
    all_data = []
    for ep in range(n_episodes):
        obs = env.reset()
        base = _get_base_env(env)
        qreb_init = base.Qreb if base is not None else None
        ep_data = {
            "Qreb_init": qreb_init,
            "rewards": [], "Qreb": [], "Tbottom": [], "Ttop": [], "delta_T": [],
        }
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_vec, infos = env.step(action)
            done = done_vec[0]
            info = infos[0]
            ep_data["rewards"].append(float(reward[0]))
            ep_data["Qreb"].append(info["Qreb"])
            ep_data["Tbottom"].append(info["Tbottom"])
            ep_data["Ttop"].append(info["Ttop"])
            ep_data["delta_T"].append(info["delta_T"])
        all_data.append(ep_data)
    return all_data


def load_episode_log(run_dir):
    path = os.path.join(run_dir, "episode_log.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _moving_average(ep_nums, values, window=50):
    arr = np.array(values, dtype=float)
    ep_arr = np.array(ep_nums)
    if len(arr) == 0:
        return np.array([]), np.array([])
    w = min(window, max(1, len(arr) // 5))
    ma = np.convolve(arr, np.ones(w) / w, mode="valid")
    return ep_arr[w - 1:], ma


def _generate_deltaT_curves(reward_type, n_curves=5):
    qreb_vals = np.linspace(QREB_MIN, QREB_MAX, 80)
    curves = []
    for k in range(n_curves):
        env = StripperSurrogateEnv(reward_type=reward_type, seed=k * 100)
        env._rng = np.random.default_rng(k * 100)
        env.Qfeed = float(env._rng.uniform(env.input_mins[0], env.input_maxs[0]))
        env.Tfeed = float(env._rng.uniform(env.input_mins[1], env.input_maxs[1]))
        env.cH2S = float(env._rng.uniform(env.input_mins[2], env.input_maxs[2]))
        env.cNH3 = float(env._rng.uniform(env.input_mins[3], env.input_maxs[3]))
        env.Pcolumn = float(env._rng.uniform(env.input_mins[5], env.input_maxs[5]))
        env.Ecolumn = float(env._rng.uniform(env.input_mins[6], env.input_maxs[6]))
        dt_curve = []
        for q in qreb_vals:
            env.Qreb = float(q)
            env._predict()
            dt_curve.append(env.Tbottom - env.Ttop)
        curves.append(dt_curve)
    return qreb_vals, curves


# ── Viz 1: Q̇reb_final vs episódio (convergência) ────────────────────────────

def plot_qreb_convergence(episode_log, all_data, reward_type, save_dir):
    if episode_log is not None:
        pairs = [(e["episode"], e["Qreb_final"]) for e in episode_log
                 if e.get("Qreb_final") is not None]
        source = "treino"
    else:
        pairs = [(i, ep["Qreb"][-1]) for i, ep in enumerate(all_data)]
        source = "eval"
    if not pairs:
        return None
    ep_nums, qreb_finals = zip(*pairs)
    ep_nums, qreb_finals = list(ep_nums), list(qreb_finals)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(ep_nums, qreb_finals, alpha=0.25, s=6, color="steelblue", label="Q̇reb final")
    ma_ep, ma_vals = _moving_average(ep_nums, qreb_finals, window=50)
    if len(ma_ep) > 0:
        ax.plot(ma_ep, ma_vals, "r-", linewidth=2, label="Média móvel (50 ep)")
    ax.axhline(QREB_MIN, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(QREB_MAX, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Q̇reb final (Gcal/h)")
    ax.set_title(f"Viz 1 — Convergência Q̇reb final vs episódio · Reward {reward_type} [{source}]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"viz1_qreb_convergence_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Viz 2: Heatmap Q̇reb_init → Q̇reb_final ──────────────────────────────────

def plot_qreb_init_vs_final(episode_log, all_data, reward_type, save_dir):
    if episode_log is not None:
        pairs = [(e["Qreb_init"], e["Qreb_final"]) for e in episode_log
                 if e.get("Qreb_init") is not None and e.get("Qreb_final") is not None]
        source = "treino"
    else:
        pairs = [(ep["Qreb_init"], ep["Qreb"][-1]) for ep in all_data
                 if ep.get("Qreb_init") is not None]
        source = "eval"
    if not pairs:
        return None
    inits, finals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(list(inits), list(finals), bins=30, cmap="Blues")
    plt.colorbar(h[3], ax=ax, label="Contagem")
    lims = [QREB_MIN, QREB_MAX]
    ax.plot(lims, lims, "r--", alpha=0.5, linewidth=1.2, label="Diagonal (sem mudança)")
    ax.set_xlabel("Q̇reb inicial (Gcal/h)")
    ax.set_ylabel("Q̇reb final (Gcal/h)")
    ax.set_title(f"Viz 2 — Q̇reb inicial → Q̇reb final · Reward {reward_type} [{source}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"viz2_qreb_init_vs_final_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Viz 3: Trajetórias intra-episódio: início vs fim do treino ───────────────

def plot_intra_trajectories(episode_log, reward_type, save_dir):
    if episode_log is None:
        return None
    early = [e for e in episode_log if e.get("trajectory") is not None and e["episode"] < 10][:5]
    late = [e for e in reversed(episode_log) if e.get("trajectory") is not None][:5]
    late = list(reversed(late))
    if not early and not late:
        return None

    n_early = max(len(early), 1)
    n_late = max(len(late), 1)
    colors_early = plt.cm.Blues(np.linspace(0.45, 0.9, n_early))
    colors_late = plt.cm.Reds(np.linspace(0.45, 0.9, n_late))

    plot_specs = [
        ("Qreb",    "Q̇reb (Gcal/h)", "Q̇reb vs passo"),
        ("delta_T", "ΔT (°C)",        "ΔT vs passo"),
        ("rewards", "Reward",          "Reward vs passo"),
    ]
    letters = ["a", "b", "c"]

    # Individual figures
    for letter, (key, ylabel, title) in zip(letters, plot_specs):
        fig_ind, ax = plt.subplots(figsize=(7, 5))
        for ci, ep in enumerate(early):
            vals = ep["trajectory"][key]
            ax.plot(range(1, len(vals) + 1), vals, color=colors_early[ci],
                    linewidth=1.3, alpha=0.85)
        for ci, ep in enumerate(late):
            vals = ep["trajectory"][key]
            ax.plot(range(1, len(vals) + 1), vals, color=colors_late[ci],
                    linewidth=1.3, alpha=0.85, linestyle="--")
        ax.set_xlabel("Passo")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        handles = [
            plt.Line2D([0], [0], color=colors_early[-1], linewidth=1.8, label="Início treino"),
            plt.Line2D([0], [0], color=colors_late[-1], linewidth=1.8,
                       linestyle="--", label="Final treino"),
        ]
        ax.legend(handles=handles, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"viz3{letter}_trajectories_early_late_reward{reward_type}.png"), dpi=150)
        plt.close()

    # Combined figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Viz 3 — Trajetórias intra-episódio (início vs fim treino) · Reward {reward_type}",
                 fontsize=12, fontweight="bold")

    for ax_idx, (key, ylabel, title) in enumerate(plot_specs):
        ax = axes[ax_idx]
        for ci, ep in enumerate(early):
            vals = ep["trajectory"][key]
            ax.plot(range(1, len(vals) + 1), vals, color=colors_early[ci],
                    linewidth=1.3, alpha=0.85)
        for ci, ep in enumerate(late):
            vals = ep["trajectory"][key]
            ax.plot(range(1, len(vals) + 1), vals, color=colors_late[ci],
                    linewidth=1.3, alpha=0.85, linestyle="--")
        ax.set_xlabel("Passo")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            handles = [
                plt.Line2D([0], [0], color=colors_early[-1], linewidth=1.8, label="Início treino"),
                plt.Line2D([0], [0], color=colors_late[-1], linewidth=1.8,
                           linestyle="--", label="Final treino"),
            ]
            ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"viz3_trajectories_early_late_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Viz 4: Mapa físico ΔT vs Q̇reb + pontos finais ───────────────────────────

def plot_physical_map(all_data, reward_type, save_dir):
    try:
        qreb_vals, curves = _generate_deltaT_curves(reward_type, n_curves=5)
    except Exception as exc:
        print(f"  [warn] Não foi possível gerar curvas físicas: {exc}")
        return None

    final_qreb = [ep["Qreb"][-1] for ep in all_data]
    final_dt = [ep["delta_T"][-1] for ep in all_data]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, curve in enumerate(curves):
        ax.plot(qreb_vals, curve, color="steelblue", alpha=0.35, linewidth=1.3,
                label="Curva surrogate" if i == 0 else "")
    ax.scatter(final_qreb, final_dt, color="red", s=22, alpha=0.55, zorder=5,
               label="Q̇reb final (eval)")
    med = float(np.median(final_qreb))
    ax.axvline(med, color="darkred", linestyle="--", linewidth=1.5,
               label=f"Mediana final: {med:.3f}")
    ax.set_xlabel("Q̇reb (Gcal/h)")
    ax.set_ylabel("ΔT = Tbottom − Ttop (°C)")
    ax.set_title(f"Viz 4 — Mapa físico ΔT vs Q̇reb · Reward {reward_type}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"viz4_physical_map_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Viz 5: Histogramas finais Q̇reb e ΔT ─────────────────────────────────────

def plot_final_histograms(all_data, reward_type, save_dir):
    final_qreb = [ep["Qreb"][-1] for ep in all_data]
    final_dt = [ep["delta_T"][-1] for ep in all_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Viz 5 — Histogramas finais · Reward {reward_type} ({len(all_data)} episódios)",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.hist(final_qreb, bins=25, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(np.mean(final_qreb), color="red", linestyle="--", linewidth=2,
               label=f"Média: {np.mean(final_qreb):.3f}")
    ax.axvline(np.median(final_qreb), color="orange", linestyle=":", linewidth=2,
               label=f"Mediana: {np.median(final_qreb):.3f}")
    ax.set_xlabel("Q̇reb final (Gcal/h)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição Q̇reb final")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(final_dt, bins=25, edgecolor="black", alpha=0.7, color="darkorange")
    ax.axvline(np.mean(final_dt), color="red", linestyle="--", linewidth=2,
               label=f"Média: {np.mean(final_dt):.3f}")
    ax.axvline(np.median(final_dt), color="blue", linestyle=":", linewidth=2,
               label=f"Mediana: {np.median(final_dt):.3f}")
    ax.set_xlabel("ΔT final (°C)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição ΔT final")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"viz5_final_histograms_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Viz 6: Erro absoluto vs episódio ─────────────────────────────────────────

def plot_error_vs_episode(episode_log, all_data, reward_type, save_dir, qreb_star=None):
    if episode_log is not None:
        pairs = [(e["episode"], e["Qreb_final"]) for e in episode_log
                 if e.get("Qreb_final") is not None]
        source = "treino"
    else:
        pairs = [(i, ep["Qreb"][-1]) for i, ep in enumerate(all_data)]
        source = "eval"
    if not pairs:
        return None
    ep_nums, qreb_finals = zip(*pairs)
    ep_nums, qreb_finals = list(ep_nums), list(qreb_finals)

    estimated = qreb_star is None
    if estimated:
        tail = qreb_finals[max(0, len(qreb_finals) * 4 // 5):]
        qreb_star = float(np.median(tail))

    errors = [abs(q - qreb_star) for q in qreb_finals]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(ep_nums, errors, alpha=0.25, s=6, color="purple", label="Erro absoluto")
    ma_ep, ma_vals = _moving_average(ep_nums, errors, window=50)
    if len(ma_ep) > 0:
        ax.plot(ma_ep, ma_vals, "r-", linewidth=2, label="Média móvel (50 ep)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    lbl = f"Q̇reb* = {qreb_star:.3f} {'(estimado — mediana dos últimos 20%)' if estimated else '(referência)'}"
    ax.set_xlabel("Episódio")
    ax.set_ylabel("|Q̇reb_final − Q̇reb*| (Gcal/h)")
    ax.set_title(f"Viz 6 — Erro absoluto vs episódio · Reward {reward_type} [{source}]\n{lbl}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"viz6_error_vs_episode_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Viz 7: Scatter reward_mean vs Q̇reb_final ─────────────────────────────────

def plot_reward_vs_qreb(episode_log, all_data, reward_type, save_dir):
    if episode_log is not None:
        rows = [(e["Qreb_final"], e["reward_mean"], e["episode"]) for e in episode_log
                if e.get("Qreb_final") is not None and e.get("reward_mean") is not None]
        source = "treino"
    else:
        rows = [(ep["Qreb"][-1], float(np.mean(ep["rewards"])), i)
                for i, ep in enumerate(all_data)]
        source = "eval"
    if not rows:
        return None
    qreb_finals, reward_means, ep_idx = zip(*rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(list(qreb_finals), list(reward_means), alpha=0.3, s=8,
                    c=list(ep_idx), cmap="viridis")
    plt.colorbar(sc, ax=ax, label="Episódio")
    ax.set_xlabel("Q̇reb final (Gcal/h)")
    ax.set_ylabel("Reward médio por episódio")
    ax.set_title(f"Viz 7 — Reward médio vs Q̇reb final · Reward {reward_type} [{source}]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"viz7_reward_vs_qreb_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Viz 8: Variância intra-episódio vs episódio ───────────────────────────────

def plot_variance_evolution(episode_log, all_data, reward_type, save_dir):
    if episode_log is not None:
        pairs = [(e["episode"], e["var_Qreb"]) for e in episode_log
                 if e.get("var_Qreb") is not None]
        source = "treino"
    else:
        pairs = [(i, float(np.var(ep["Qreb"]))) for i, ep in enumerate(all_data)]
        source = "eval"
    if not pairs:
        return None
    ep_nums, variances = zip(*pairs)
    ep_nums, variances = list(ep_nums), list(variances)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(ep_nums, variances, alpha=0.25, s=6, color="teal", label="Var(Q̇reb)")
    ma_ep, ma_vals = _moving_average(ep_nums, variances, window=50)
    if len(ma_ep) > 0:
        ax.plot(ma_ep, ma_vals, "r-", linewidth=2, label="Média móvel (50 ep)")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Var(Q̇reb) intra-episódio")
    ax.set_title(f"Viz 8 — Variância intra-episódio Q̇reb · Reward {reward_type} [{source}]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"viz8_variance_evolution_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Viz 9: Trajetórias por faixa de inicialização ────────────────────────────

def run_episodes_forced_band(model, env, band_name, band_range, n=3):
    _BAND_SEED_OFFSET = {"low": 0, "mid": 100, "high": 200}
    offset = _BAND_SEED_OFFSET.get(band_name, 0)
    all_data = []
    for ep in range(n):
        obs = env.reset()
        base = _get_base_env(env)
        if base is not None:
            rng = np.random.default_rng(SEED + offset + ep * 7)
            base.Qreb = float(rng.uniform(band_range[0], band_range[1]))
            base._predict()
        qreb_init = base.Qreb if base is not None else None
        ep_data = {
            "Qreb_init": qreb_init,
            "band": band_name,
            "rewards": [float("nan")],
            "Qreb": [qreb_init] if qreb_init is not None else [],
            "Tbottom": [base.Tbottom] if base is not None else [],
            "Ttop": [base.Ttop] if base is not None else [],
            "delta_T": [base.Tbottom - base.Ttop] if base is not None else [],
        }
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, infos = env.step(action)
            done = done_vec[0]
            info = infos[0]
            ep_data["rewards"].append(float(reward[0]))
            ep_data["Qreb"].append(info["Qreb"])
            ep_data["Tbottom"].append(info["Tbottom"])
            ep_data["Ttop"].append(info["Ttop"])
            ep_data["delta_T"].append(info["delta_T"])
        all_data.append(ep_data)
    return all_data


def plot_trajectories_by_band(model, env, reward_type, save_dir):
    import matplotlib.gridspec as gridspec

    bands = [
        ("low",  QREB_INIT_LOW,  plt.cm.Blues,  "Low"),
        ("mid",  QREB_INIT_MID,  plt.cm.Greens, "Mid"),
        ("high", QREB_INIT_HIGH, plt.cm.Reds,   "High"),
    ]
    linestyles = ["-", "--", ":"]

    band_data = {}
    for band_name, band_range, _, _ in bands:
        band_data[band_name] = run_episodes_forced_band(model, env, band_name, band_range, n=3)

    keys = [
        ("Qreb",    "Q̇reb (Gcal/h)", "Q̇reb vs passo"),
        ("delta_T", "ΔT (°C)",        "ΔT vs passo"),
        ("Tbottom", "Tbottom (°C)",   "Tbottom vs passo"),
        ("Ttop",    "Ttop (°C)",      "Ttop vs passo"),
        ("rewards", "Reward",          "Reward vs passo"),
    ]
    letters = ["a", "b", "c", "d", "e"]

    # Individual figures
    for letter, (key, ylabel, title) in zip(letters, keys):
        fig_ind, ax = plt.subplots(figsize=(7, 5))
        for band_name, band_range, cmap, label in bands:
            colors = cmap(np.linspace(0.45, 0.85, 3))
            for ei, ep in enumerate(band_data[band_name]):
                qinit = ep["Qreb_init"]
                lbl = f"{label} ep{ei + 1} (Q0={qinit:.2f})" if qinit is not None else f"{label} ep{ei + 1}"
                vals = ep[key]
                ax.plot(range(0, len(vals)), vals,
                        color=colors[ei], linewidth=1.4,
                        linestyle=linestyles[ei], alpha=0.87,
                        label=lbl)
        ax.set_xlabel("Passo (0 = estado inicial)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=1)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"viz9{letter}_trajectories_by_band_reward{reward_type}.png"), dpi=150)
        plt.close()

    # Combined figure — 3 plots on top, 2 centered on bottom
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.35)
    ax_positions = [
        gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
        gs[1, 1:3], gs[1, 3:5],
    ]
    axes = [fig.add_subplot(pos) for pos in ax_positions]
    fig.suptitle(
        f"Viz 9 — Trajetórias por faixa de inicialização · Reward {reward_type}",
        fontsize=12, fontweight="bold",
    )

    for ax_idx, (key, ylabel, title) in enumerate(keys):
        ax = axes[ax_idx]
        for band_name, band_range, cmap, label in bands:
            colors = cmap(np.linspace(0.45, 0.85, 3))
            for ei, ep in enumerate(band_data[band_name]):
                qinit = ep["Qreb_init"]
                lbl = f"{label} ep{ei + 1} (Q0={qinit:.2f})" if qinit is not None else f"{label} ep{ei + 1}"
                vals = ep[key]
                ax.plot(range(0, len(vals)), vals,
                        color=colors[ei], linewidth=1.4,
                        linestyle=linestyles[ei], alpha=0.87,
                        label=lbl)
        ax.set_xlabel("Passo (0 = estado inicial)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=1)

    fname = os.path.join(save_dir, f"viz9_trajectories_by_band_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# ── Funções originais mantidas ────────────────────────────────────────────────

def plot_episode_trajectory(ep_data, episode_idx, reward_type, save_dir):
    steps = range(1, len(ep_data["Qreb"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Episode {episode_idx} — Reward {reward_type}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, ep_data["Qreb"], "b-o", markersize=3, linewidth=1.5)
    ax.axhline(QREB_MIN, color="r", linestyle="--", alpha=0.5, label="Qreb min")
    ax.axhline(QREB_MAX, color="r", linestyle="--", alpha=0.5, label="Qreb max")
    ax.set_ylabel("Qreb (Gcal/h)")
    ax.set_xlabel("Step")
    ax.set_title("Reboiler Duty")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, ep_data["Tbottom"], "r-o", markersize=3, linewidth=1.5, label="Tbottom")
    ax.plot(steps, ep_data["Ttop"], "b-s", markersize=3, linewidth=1.5, label="Ttop")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Step")
    ax.set_title("Temperatures")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, ep_data["delta_T"], "g-o", markersize=3, linewidth=1.5)
    ax.set_ylabel("ΔT = Tbottom − Ttop (°C)")
    ax.set_xlabel("Step")
    ax.set_title("Thermal Gradient")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, ep_data["rewards"], "m-o", markersize=3, linewidth=1.5)
    ax.set_ylabel("Reward")
    ax.set_xlabel("Step")
    ax.set_title("Step Reward")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"trajectory_reward{reward_type}_ep{episode_idx}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_training_curve(log_dir, reward_type, save_dir):
    eval_file = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(eval_file):
        return None
    data = np.load(eval_file)
    timesteps = data["timesteps"]
    results = data["results"]
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, mean_rewards, "b-", linewidth=2, label="Mean eval reward")
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                    alpha=0.2, color="blue")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Training Curve — DDPG Reward {reward_type}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"training_curve_reward{reward_type}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_comparison(data_a, data_b, save_dir):
    if data_a is None or data_b is None:
        return

    def _stats(all_data):
        returns = [sum(ep["rewards"]) for ep in all_data]
        final_qreb = [ep["Qreb"][-1] for ep in all_data]
        final_dt = [ep["delta_T"][-1] for ep in all_data]
        return returns, final_qreb, final_dt

    ret_a, qreb_a, dt_a = _stats(data_a)
    ret_b, qreb_b, dt_b = _stats(data_b)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.boxplot([ret_a, ret_b], labels=["Reward A", "Reward B"])
    ax.set_ylabel("Episode Return")
    ax.set_title("Episode Returns")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.boxplot([qreb_a, qreb_b], labels=["Reward A", "Reward B"])
    ax.set_ylabel("Final Qreb (Gcal/h)")
    ax.set_title("Final Reboiler Duty")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.boxplot([dt_a, dt_b], labels=["Reward A", "Reward B"])
    ax.set_ylabel("Final ΔT (°C)")
    ax.set_title("Final Thermal Gradient")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Comparison: Reward A vs Reward B", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(save_dir, "comparison_A_vs_B.png")
    plt.savefig(fname, dpi=150)
    plt.close()


def evaluate_run(run_dir, reward_type, n_episodes):
    print(f"\nEvaluating: {run_dir} (Reward {reward_type})")
    model, env = load_model(run_dir, reward_type)
    all_data = run_episodes(model, env, n_episodes)

    returns = [sum(ep["rewards"]) for ep in all_data]
    final_qreb = [ep["Qreb"][-1] for ep in all_data]
    final_dt = [ep["delta_T"][-1] for ep in all_data]

    stats = {
        "run_dir": run_dir,
        "reward_type": reward_type,
        "n_episodes": n_episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "final_qreb_mean": float(np.mean(final_qreb)),
        "final_qreb_std": float(np.std(final_qreb)),
        "final_deltaT_mean": float(np.mean(final_dt)),
        "final_deltaT_std": float(np.std(final_dt)),
    }

    print(f"  Return:     {stats['return_mean']:.3f} ± {stats['return_std']:.3f}")
    print(f"  Final Qreb: {stats['final_qreb_mean']:.3f} ± {stats['final_qreb_std']:.3f}")
    print(f"  Final ΔT:   {stats['final_deltaT_mean']:.3f} ± {stats['final_deltaT_std']:.3f}")

    run_name = os.path.basename(run_dir)
    fig_dir = os.path.join(FIGURES_DIR, run_name)
    os.makedirs(fig_dir, exist_ok=True)

    episode_log = load_episode_log(run_dir)
    if episode_log is not None:
        print(f"  Episode log: {len(episode_log)} episódios de treino carregados")
    else:
        print("  Episode log não encontrado — vizualizações de treino usarão dados de eval")

    plot_trajectories_by_band(model, env, reward_type, fig_dir)

    plot_qreb_convergence(episode_log, all_data, reward_type, fig_dir)
    plot_qreb_init_vs_final(episode_log, all_data, reward_type, fig_dir)
    plot_intra_trajectories(episode_log, reward_type, fig_dir)
    plot_physical_map(all_data, reward_type, fig_dir)
    plot_final_histograms(all_data, reward_type, fig_dir)
    plot_error_vs_episode(episode_log, all_data, reward_type, fig_dir)
    plot_reward_vs_qreb(episode_log, all_data, reward_type, fig_dir)
    plot_variance_evolution(episode_log, all_data, reward_type, fig_dir)

    log_subdir = os.path.join(LOGS_DIR, run_name)
    plot_training_curve(log_subdir, reward_type, fig_dir)

    stats_path = os.path.join(fig_dir, f"eval_stats_reward{reward_type}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return all_data, stats


def list_available_runs():
    pattern = os.path.join(MODELS_DIR, "DDPG_reward*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print("No runs found in", MODELS_DIR)
        return []
    print(f"Available runs in {MODELS_DIR}:")
    for m in matches:
        name = os.path.basename(m)
        files = os.listdir(m)
        model_file = "best_model.zip" if "best_model.zip" in files else (
            "final_model.zip" if "final_model.zip" in files else "no model found")
        ep_log = "episode_log.json" if "episode_log.json" in files else "no episode log"
        print(f"  {name}  [{model_file}]  [{ep_log}]")
    return matches


def find_run(run_arg):
    pattern = os.path.join(MODELS_DIR, "DDPG_reward*")
    matches = sorted(glob.glob(pattern))
    hits = []
    for m in matches:
        parts = os.path.basename(m).split("_")
        if len(parts) >= 3 and parts[2] == run_arg:
            hits.append(m)
    return hits


def infer_reward_type(run_dir):
    name = os.path.basename(run_dir)
    if "rewardA" in name:
        return "A"
    if "rewardB" in name:
        return "B"
    return None


def find_latest_runs():
    runs = {}
    for rtype in ["A", "B"]:
        pattern = os.path.join(MODELS_DIR, f"DDPG_reward{rtype}_*")
        matches = sorted(glob.glob(pattern))
        if matches:
            runs[rtype] = matches[-1]
    return runs


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if RUN is not None:
        hits = find_run(RUN)
        if not hits:
            print(f"Nenhum run encontrado com tag '{RUN}' em {MODELS_DIR}")
            list_available_runs()
            sys.exit(1)
        for run_dir in hits:
            reward = infer_reward_type(run_dir)
            if not reward:
                print(f"Nao foi possivel inferir o reward type de '{os.path.basename(run_dir)}', ignorando.")
                continue
            evaluate_run(run_dir, reward, N_EPISODES)

    else:
        runs = find_latest_runs()
        if not runs:
            print("Nenhum modelo encontrado. Execute train_ddpg.py primeiro.")
            sys.exit(1)
        for rtype, rdir in runs.items():
            evaluate_run(rdir, rtype, N_EPISODES)


if __name__ == "__main__":
    main()
