import os
import sys
import glob
import csv

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# ── Adjustable parameters ─────────────────────────────────────────────────────
RUN    = "opt02"   # deve corresponder ao RUN_TAG usado em 12-agentRL-DDPG-Opt
REWARD = "A"
N_EPISODES_PER_BAND      = 10
SEED                     = 42
QREB_CONVERGENCE_THRESHOLD = 0.01
# ─────────────────────────────────────────────────────────────────────────────

base_dir    = os.path.dirname(os.path.abspath(__file__))
data_dir    = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)
scripts_dir = os.path.abspath(os.path.join(base_dir, "..", "16-agentRL-DDPG-Opt2", "scripts"))
models_dir  = os.path.abspath(os.path.join(base_dir, "..", "16-agentRL-DDPG-Opt2", "models"))
sys.path.insert(0, scripts_dir)

from config import (
    QREB_INIT_LOW, QREB_INIT_MID, QREB_INIT_HIGH,
    INPUT_MINS, INPUT_MAXS, SEED as CONFIG_SEED,
)
from surrogate_env import StripperSurrogateEnv

EPISODES_RESULTS_FILE = os.path.join(data_dir, f"episodes_results_{RUN}_{REWARD}.csv")
BANDS = ["low", "mid", "high"]
BAND_RANGES = {
    "low":  QREB_INIT_LOW,
    "mid":  QREB_INIT_MID,
    "high": QREB_INIT_HIGH,
}


def find_model_dir(models_dir, reward, run):
    """Procura o modelo DDPG Optuna: DDPG_opt_best_{reward}_{run}_*"""
    pattern = os.path.join(models_dir, f"DDPG_opt_best_{reward}_{run}_*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No model directory found matching pattern: {pattern}\n"
            f"Execute primeiro train_ddpg_optuna.py com RUN_TAG='{run}'."
        )
    return sorted(matches)[-1]


def load_model(run_dir, reward_type):
    best_path  = os.path.join(run_dir, "best_model.zip")
    final_path = os.path.join(run_dir, "final_model.zip")
    model_path = best_path if os.path.exists(best_path) else final_path

    vec_norm_path = os.path.join(run_dir, "vec_normalize.pkl")

    env = DummyVecEnv([lambda: Monitor(StripperSurrogateEnv(reward_type=reward_type, seed=CONFIG_SEED + 9999))])
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training    = False
        env.norm_reward = False

    model = DDPG.load(model_path, env=env)
    return model, env


def get_base_env(env):
    try:
        return env.venv.envs[0].env
    except AttributeError:
        try:
            return env.envs[0].env
        except AttributeError:
            return None


def run_episode_forced_band(model, env, base, rng, band):
    env.reset()

    base.Qfeed   = float(rng.uniform(INPUT_MINS[0], INPUT_MAXS[0]))
    base.Tfeed   = float(rng.uniform(INPUT_MINS[1], INPUT_MAXS[1]))
    base.cH2S    = float(rng.uniform(INPUT_MINS[2], INPUT_MAXS[2]))
    base.cNH3    = float(rng.uniform(INPUT_MINS[3], INPUT_MAXS[3]))
    base.Pcolumn = float(rng.uniform(INPUT_MINS[5], INPUT_MAXS[5]))
    base.Ecolumn = float(rng.uniform(INPUT_MINS[6], INPUT_MAXS[6]))
    base.Qreb    = float(rng.uniform(BAND_RANGES[band][0], BAND_RANGES[band][1]))
    base._predict()
    base._step_count = 0
    base._tbottom_history.clear()
    base._tbottom_history.append(base.Tbottom)

    raw_obs = base._get_obs().reshape(1, -1).astype(np.float32)
    if isinstance(env, VecNormalize):
        obs = env.normalize_obs(raw_obs)
    else:
        obs = raw_obs

    initial = {
        "Qfeed":   base.Qfeed,
        "Tfeed":   base.Tfeed,
        "cH2S":    base.cH2S,
        "cNH3":    base.cNH3,
        "Qreb":    base.Qreb,
        "Pcolumn": base.Pcolumn,
        "Ecolumn": base.Ecolumn,
    }

    done = False
    qreb_history = [initial["Qreb"]]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_vec, infos = env.step(action)
        done = done_vec[0]
        info = infos[0]
        qreb_history.append(info["Qreb"])

    last_change_step = 0
    for i in range(1, len(qreb_history)):
        if abs(qreb_history[i] - qreb_history[i - 1]) > QREB_CONVERGENCE_THRESHOLD:
            last_change_step = i

    return {
        "Qfeed":        initial["Qfeed"],
        "Tfeed":        initial["Tfeed"],
        "cH2S":         initial["cH2S"],
        "cNH3":         initial["cNH3"],
        "Qreb_init":    initial["Qreb"],
        "Pcolumn":      initial["Pcolumn"],
        "Ecolumn":      initial["Ecolumn"],
        "Qreb_final":   info["Qreb"],
        "n_iter_final": last_change_step,
    }


def main():
    model_dir = find_model_dir(models_dir, REWARD, RUN)
    print(f"Using model directory: {model_dir}")

    model, env = load_model(model_dir, REWARD)
    base = get_base_env(env)
    if base is None:
        raise RuntimeError("Could not access base environment.")

    rng = np.random.default_rng(SEED)

    all_episodes = []
    episode_id   = 1
    for band in BANDS:
        print(f"Running {N_EPISODES_PER_BAND} episodes for band '{band}'...")
        for _ in range(N_EPISODES_PER_BAND):
            result = run_episode_forced_band(model, env, base, rng, band)
            result["episode_id"] = episode_id
            result["band"]       = band
            all_episodes.append(result)
            print(
                f"  ep {episode_id:>3d} | band={band} | "
                f"Qreb_init={result['Qreb_init']:.4f} -> Qreb_final={result['Qreb_final']:.4f} | "
                f"n_iter={result['n_iter_final']}"
            )
            episode_id += 1

    episodes_fieldnames = [
        "episode_id", "band",
        "Qfeed", "Tfeed", "cH2S", "cNH3",
        "Qreb_init", "Pcolumn", "Ecolumn",
        "Qreb_final", "n_iter_final",
    ]
    with open(EPISODES_RESULTS_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episodes_fieldnames)
        writer.writeheader()
        for ep in all_episodes:
            writer.writerow({k: ep[k] for k in episodes_fieldnames})
    print(f"Episodes results saved to: {EPISODES_RESULTS_FILE}")

    sim_fieldnames = ["case_id", "Qfeed", "Tfeed", "cH2S", "cNH3", "Qreb", "Pcolumn", "Ecolumn", "Qreb_final"]
    for band in BANDS:
        band_episodes = [ep for ep in all_episodes if ep["band"] == band]
        sim_file = os.path.join(data_dir, f"sim_points_{RUN}_{REWARD}_{band}.csv")
        with open(sim_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sim_fieldnames)
            writer.writeheader()
            for ep in band_episodes:
                writer.writerow({
                    "case_id":    ep["episode_id"],
                    "Qfeed":      ep["Qfeed"],
                    "Tfeed":      ep["Tfeed"],
                    "cH2S":       ep["cH2S"],
                    "cNH3":       ep["cNH3"],
                    "Qreb":       ep["Qreb_init"],
                    "Pcolumn":    ep["Pcolumn"],
                    "Ecolumn":    ep["Ecolumn"],
                    "Qreb_final": ep["Qreb_final"],
                })
        print(f"Aspen input CSV saved to: {sim_file}")


if __name__ == "__main__":
    main()
