"""
train_ppo_optuna.py
Otimização de hiperparâmetros do agente PPO com Optuna.

Fase 1 — Estudo Optuna:
    Executa N_TRIALS trials, cada um treinando por OPTUNA_TIMESTEPS.
    Métrica: best_mean_reward do EvalCallback.

Fase 2 — Retreino do melhor trial:
    Retreina com os melhores hiperparâmetros por TOTAL_TIMESTEPS.
    Salva o modelo final em models/PPO_opt_best_{REWARD}_{RUN_TAG}_{timestamp}/.

Uso:
    .venv\\Scripts\\python.exe 10-agentRL-PPO-Opt/scripts/train_ppo_optuna.py
"""

import os
import sys
import json
import logging
from collections import deque
from datetime import datetime

import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from config import (
    EVAL_FREQ,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    N_EVAL_EPISODES,
    N_TRIALS,
    OPTUNA_TIMESTEPS,
    PPO_SEARCH_SPACE,
    REWARD_SEARCH_SPACE_A,
    REWARD_TYPES,
    RUN_TAG,
    SEED,
    STUDIES_DIR,
    TOTAL_TIMESTEPS,
)
from surrogate_env import StripperSurrogateEnv

LOG_FILE = os.path.join(LOGS_DIR, f"train_ppo_optuna-{RUN_TAG}.log")


os.makedirs(LOGS_DIR, exist_ok=True)


class FlushableFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    handlers=[
        FlushableFileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Silencia logs do Optuna para não poluir o arquivo de log principal
optuna.logging.set_verbosity(optuna.logging.WARNING)


TRIAL_LOG_INTERVAL = 10_000   # a cada quantos steps exibir progresso do trial


# ── Callbacks ─────────────────────────────────────────────────────────────────

class TrialProgressCallback(BaseCallback):
    """Imprime progresso leve a cada TRIAL_LOG_INTERVAL steps durante um trial."""

    def __init__(self, trial_num, total_timesteps, log_interval=TRIAL_LOG_INTERVAL, verbose=0):
        super().__init__(verbose)
        self.trial_num       = trial_num
        self.total_timesteps = total_timesteps
        self.log_interval    = log_interval
        self.episode_rewards = []
        self.current_rewards = None

    def _on_training_start(self):
        self.current_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        self.current_rewards += self.locals["rewards"]
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_rewards.append(self.current_rewards[i])
                self.current_rewards[i] = 0.0

        if self.num_timesteps % self.log_interval < self.training_env.num_envs:
            pct    = 100 * self.num_timesteps / self.total_timesteps
            recent = self.episode_rewards[-20:] if self.episode_rewards else []
            mean_r = float(np.mean(recent)) if recent else float("nan")
            print(
                f"  [Trial {self.trial_num:03d}] "
                f"{self.num_timesteps:>7d}/{self.total_timesteps} ({pct:5.1f}%) | "
                f"ep={len(self.episode_rewards):>4d} | "
                f"mean_r(last 20)={mean_r:>8.3f}",
                flush=True,
            )
        return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env(reward_type, seed, reward_weights_a=None):
    def _init():
        env = StripperSurrogateEnv(
            reward_type=reward_type,
            seed=seed,
            reward_weights_a=reward_weights_a,
        )
        return Monitor(env)
    return _init


def suggest_ppo_params(trial):
    """Sugere hiperparâmetros PPO a partir do espaço de busca definido em config."""
    n_steps = trial.suggest_categorical(
        "n_steps", PPO_SEARCH_SPACE["n_steps"]["choices"]
    )
    # batch_size deve dividir n_steps (com 1 env: n_steps % batch_size == 0)
    valid_batch = [
        x for x in PPO_SEARCH_SPACE["batch_size"]["choices"] if n_steps % x == 0
    ]
    batch_size = trial.suggest_categorical("batch_size", valid_batch)

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            PPO_SEARCH_SPACE["learning_rate"]["low"],
            PPO_SEARCH_SPACE["learning_rate"]["high"],
            log=True,
        ),
        "n_steps":       n_steps,
        "batch_size":    batch_size,
        "n_epochs":      trial.suggest_int(
            "n_epochs",
            PPO_SEARCH_SPACE["n_epochs"]["low"],
            PPO_SEARCH_SPACE["n_epochs"]["high"],
        ),
        "gamma":         trial.suggest_float(
            "gamma",
            PPO_SEARCH_SPACE["gamma"]["low"],
            PPO_SEARCH_SPACE["gamma"]["high"],
        ),
        "gae_lambda":    trial.suggest_float(
            "gae_lambda",
            PPO_SEARCH_SPACE["gae_lambda"]["low"],
            PPO_SEARCH_SPACE["gae_lambda"]["high"],
        ),
        "clip_range":    trial.suggest_float(
            "clip_range",
            PPO_SEARCH_SPACE["clip_range"]["low"],
            PPO_SEARCH_SPACE["clip_range"]["high"],
        ),
        "ent_coef":      trial.suggest_float(
            "ent_coef",
            PPO_SEARCH_SPACE["ent_coef"]["low"],
            PPO_SEARCH_SPACE["ent_coef"]["high"],
        ),
        "vf_coef":       trial.suggest_float(
            "vf_coef",
            PPO_SEARCH_SPACE["vf_coef"]["low"],
            PPO_SEARCH_SPACE["vf_coef"]["high"],
        ),
        "max_grad_norm": trial.suggest_float(
            "max_grad_norm",
            PPO_SEARCH_SPACE["max_grad_norm"]["low"],
            PPO_SEARCH_SPACE["max_grad_norm"]["high"],
        ),
    }


def suggest_reward_weights(trial, reward_type):
    """Sugere pesos da função de recompensa (incluídos na busca)."""
    if reward_type == "A":
        return {
            "w1": trial.suggest_float(
                "w1",
                REWARD_SEARCH_SPACE_A["w1"]["low"],
                REWARD_SEARCH_SPACE_A["w1"]["high"],
            ),
            "w2": trial.suggest_float(
                "w2",
                REWARD_SEARCH_SPACE_A["w2"]["low"],
                REWARD_SEARCH_SPACE_A["w2"]["high"],
            ),
            "w3": trial.suggest_float(
                "w3",
                REWARD_SEARCH_SPACE_A["w3"]["low"],
                REWARD_SEARCH_SPACE_A["w3"]["high"],
            ),
        }
    return None


# ── Trial training ────────────────────────────────────────────────────────────

def run_trial(ppo_params, reward_weights_a, reward_type, trial_num, timesteps):
    """Treina um trial e retorna (best_mean_reward, run_name)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"PPO_opt_trial{trial_num:03d}_{reward_type}_{timestamp}"

    train_env = DummyVecEnv([make_env(reward_type, SEED, reward_weights_a)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(reward_type, SEED + 1000, reward_weights_a)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )

    model_save_path = os.path.join(MODELS_DIR, run_name)
    os.makedirs(model_save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path=os.path.join(LOGS_DIR, run_name),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=0,
        tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"),
        **ppo_params,
    )

    progress_cb = TrialProgressCallback(trial_num=trial_num, total_timesteps=timesteps)
    model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_cb], progress_bar=False)

    best_reward = float(eval_callback.best_mean_reward)

    train_env.close()
    eval_env.close()

    return best_reward, run_name


# ── Optuna objective ──────────────────────────────────────────────────────────

def objective(trial, reward_type):
    ppo_params       = suggest_ppo_params(trial)
    reward_weights_a = suggest_reward_weights(trial, reward_type)

    print(
        f"\n>>> Trial {trial.number:03d} iniciando | "
        f"lr={ppo_params['learning_rate']:.2e} | "
        f"n_steps={ppo_params['n_steps']} | "
        f"batch={ppo_params['batch_size']} | "
        f"clip={ppo_params['clip_range']:.3f}",
        flush=True,
    )
    logger.info(
        f"Trial {trial.number:03d} | ppo={ppo_params} | weights={reward_weights_a}"
    )

    try:
        mean_reward, run_name = run_trial(
            ppo_params, reward_weights_a, reward_type, trial.number, OPTUNA_TIMESTEPS
        )
        logger.info(f"Trial {trial.number:03d} | best_mean_reward={mean_reward:.4f}")
        return mean_reward
    except Exception as exc:
        logger.error(f"Trial {trial.number:03d} FAILED: {exc}")
        return float("-inf")


# ── Final best-model retraining ───────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """Registra recompensas por episódio durante o retreino final."""

    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self.log_interval   = log_interval
        self.episode_rewards = []
        self.current_rewards = None

    def _on_training_start(self):
        self.current_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        self.current_rewards += self.locals["rewards"]
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_rewards.append(self.current_rewards[i])
                self.current_rewards[i] = 0.0

        if self.num_timesteps % self.log_interval < self.training_env.num_envs:
            if self.episode_rewards:
                recent = self.episode_rewards[-50:]
                logger.info(
                    f"Step {self.num_timesteps:>8d} | "
                    f"Episodes: {len(self.episode_rewards):>5d} | "
                    f"Mean reward (last 50): {np.mean(recent):>8.3f} | "
                    f"Std: {np.std(recent):>7.3f}"
                )
        return True


def retrain_best(best_params, reward_type):
    """Retreina com os melhores hiperparâmetros por TOTAL_TIMESTEPS."""
    ppo_keys    = list(PPO_SEARCH_SPACE.keys())
    reward_keys = list(REWARD_SEARCH_SPACE_A.keys())

    ppo_params = {k: best_params[k] for k in ppo_keys if k in best_params}
    reward_weights_a = None
    if reward_type == "A":
        reward_weights_a = {k: best_params[k] for k in reward_keys if k in best_params}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"PPO_opt_best_{reward_type}_{RUN_TAG}_{timestamp}"

    logger.info(f"{'=' * 60}")
    logger.info(f"Retreino final: {run_name}")
    logger.info(f"PPO params:     {ppo_params}")
    logger.info(f"Reward weights: {reward_weights_a}")
    logger.info(f"Timesteps:      {TOTAL_TIMESTEPS}")
    logger.info(f"{'=' * 60}")

    train_env = DummyVecEnv([make_env(reward_type, SEED, reward_weights_a)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(reward_type, SEED + 1000, reward_weights_a)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )

    model_save_path = os.path.join(MODELS_DIR, run_name)
    os.makedirs(model_save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path=os.path.join(LOGS_DIR, run_name),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )
    reward_logger = RewardLoggerCallback(log_interval=5000)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=0,
        tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"),
        **ppo_params,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, reward_logger],
        progress_bar=True,
    )

    final_path = os.path.join(model_save_path, "final_model")
    model.save(final_path)
    train_env.save(os.path.join(model_save_path, "vec_normalize.pkl"))

    meta = {
        "run_name":         run_name,
        "reward_type":      reward_type,
        "total_timesteps":  TOTAL_TIMESTEPS,
        "ppo_params":       ppo_params,
        "reward_weights_a": reward_weights_a,
        "seed":             SEED,
        "best_mean_reward": float(eval_callback.best_mean_reward),
        "total_episodes":   len(reward_logger.episode_rewards),
        "episode_rewards_mean": float(np.mean(reward_logger.episode_rewards[-100:])) if reward_logger.episode_rewards else 0.0,
    }
    with open(os.path.join(model_save_path, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    eval_env.close()

    logger.info(f"Modelo salvo: {run_name}")
    logger.info(f"Best eval reward: {eval_callback.best_mean_reward:.4f}")
    return run_name


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for d in [MODELS_DIR, LOGS_DIR, FIGURES_DIR, STUDIES_DIR]:
        os.makedirs(d, exist_ok=True)

    for rtype in REWARD_TYPES:
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Estudo Optuna — reward '{rtype}' | "
            f"{N_TRIALS} trials × {OPTUNA_TIMESTEPS} timesteps"
        )
        logger.info(f"{'=' * 60}")

        study = optuna.create_study(
            direction="maximize",
            study_name=f"ppo_reward{rtype}_{RUN_TAG}",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )

        study.optimize(
            lambda trial: objective(trial, rtype),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        logger.info(f"Estudo concluído — melhor trial: {study.best_trial.number}")
        logger.info(f"Melhor valor:   {study.best_value:.4f}")
        logger.info(f"Melhores params: {study.best_params}")

        # Salva resultados do estudo
        study_path = os.path.join(STUDIES_DIR, f"study_ppo_reward{rtype}_{RUN_TAG}.json")
        study_data = {
            "best_trial":  study.best_trial.number,
            "best_value":  study.best_value,
            "best_params": study.best_params,
            "n_trials":    N_TRIALS,
            "optuna_timesteps": OPTUNA_TIMESTEPS,
            "trials": [
                {
                    "number": t.number,
                    "value":  t.value,
                    "params": t.params,
                    "state":  str(t.state),
                }
                for t in study.trials
            ],
        }
        with open(study_path, "w") as f:
            json.dump(study_data, f, indent=2)
        logger.info(f"Resultados do estudo salvos em: {study_path}")

        # Retreina o melhor trial por TOTAL_TIMESTEPS
        retrain_best(study.best_params, rtype)


if __name__ == "__main__":
    main()
