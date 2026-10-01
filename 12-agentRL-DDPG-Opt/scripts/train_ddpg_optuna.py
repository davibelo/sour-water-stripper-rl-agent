"""
train_ddpg_optuna.py
Otimização de hiperparâmetros do agente DDPG com Optuna.

Fase 1 — Estudo Optuna:
    Executa N_TRIALS trials, cada um treinando por OPTUNA_TIMESTEPS.
    Métrica: best_mean_reward do EvalCallback.

Fase 2 — Retreino do melhor trial:
    Retreina com os melhores hiperparâmetros por TOTAL_TIMESTEPS.
    Salva o modelo final em models/DDPG_opt_best_{REWARD}_{RUN_TAG}_{timestamp}/.

Uso:
    .venv\\Scripts\\python.exe 12-agentRL-DDPG-Opt/scripts/train_ddpg_optuna.py
"""

import os
import sys
import json
import logging
from collections import deque
from datetime import datetime

import numpy as np
import optuna
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from config import (
    ACTION_NOISE_SEARCH_SPACE,
    DDPG_SEARCH_SPACE,
    EVAL_FREQ,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    N_EVAL_EPISODES,
    N_TRIALS,
    OPTUNA_TIMESTEPS,
    REWARD_SEARCH_SPACE_A,
    REWARD_TYPES,
    RUN_TAG,
    SEED,
    STUDIES_DIR,
    TOTAL_TIMESTEPS,
)
from surrogate_env import StripperSurrogateEnv

LOG_FILE = os.path.join(LOGS_DIR, f"train_ddpg_optuna-{RUN_TAG}.log")


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

optuna.logging.set_verbosity(optuna.logging.WARNING)


TRIAL_LOG_INTERVAL = 5_000    # DDPG é mais lento por step (TF a cada passo), intervalo menor que PPO


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


def suggest_ddpg_params(trial):
    """Sugere hiperparâmetros DDPG a partir do espaço de busca definido em config."""
    return {
        "learning_rate":   trial.suggest_float(
            "learning_rate",
            DDPG_SEARCH_SPACE["learning_rate"]["low"],
            DDPG_SEARCH_SPACE["learning_rate"]["high"],
            log=True,
        ),
        "buffer_size":     trial.suggest_categorical(
            "buffer_size",   DDPG_SEARCH_SPACE["buffer_size"]["choices"]
        ),
        "learning_starts": trial.suggest_int(
            "learning_starts",
            DDPG_SEARCH_SPACE["learning_starts"]["low"],
            DDPG_SEARCH_SPACE["learning_starts"]["high"],
        ),
        "batch_size":      trial.suggest_categorical(
            "batch_size",    DDPG_SEARCH_SPACE["batch_size"]["choices"]
        ),
        "tau":             trial.suggest_float(
            "tau",
            DDPG_SEARCH_SPACE["tau"]["low"],
            DDPG_SEARCH_SPACE["tau"]["high"],
        ),
        "gamma":           trial.suggest_float(
            "gamma",
            DDPG_SEARCH_SPACE["gamma"]["low"],
            DDPG_SEARCH_SPACE["gamma"]["high"],
        ),
        "train_freq":      trial.suggest_categorical(
            "train_freq",    DDPG_SEARCH_SPACE["train_freq"]["choices"]
        ),
        "gradient_steps":  trial.suggest_categorical(
            "gradient_steps", DDPG_SEARCH_SPACE["gradient_steps"]["choices"]
        ),
    }


def suggest_noise_sigma(trial):
    return trial.suggest_float(
        "noise_sigma",
        ACTION_NOISE_SEARCH_SPACE["sigma"]["low"],
        ACTION_NOISE_SEARCH_SPACE["sigma"]["high"],
    )


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

def run_trial(ddpg_params, noise_sigma, reward_weights_a, reward_type, trial_num, timesteps):
    """Treina um trial e retorna (best_mean_reward, run_name)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"DDPG_opt_trial{trial_num:03d}_{reward_type}_{timestamp}"

    train_env = DummyVecEnv([make_env(reward_type, SEED, reward_weights_a)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

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

    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(1),
        sigma=noise_sigma * np.ones(1),
    )

    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=0,
        tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"),
        action_noise=action_noise,
        **ddpg_params,
    )

    progress_cb = TrialProgressCallback(trial_num=trial_num, total_timesteps=timesteps)
    model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_cb], progress_bar=False)

    best_reward = float(eval_callback.best_mean_reward)

    train_env.close()
    eval_env.close()

    return best_reward, run_name


# ── Optuna objective ──────────────────────────────────────────────────────────

def objective(trial, reward_type):
    ddpg_params      = suggest_ddpg_params(trial)
    noise_sigma      = suggest_noise_sigma(trial)
    reward_weights_a = suggest_reward_weights(trial, reward_type)

    print(
        f"\n>>> Trial {trial.number:03d} iniciando | "
        f"lr={ddpg_params['learning_rate']:.2e} | "
        f"batch={ddpg_params['batch_size']} | "
        f"tau={ddpg_params['tau']:.4f} | "
        f"noise_sigma={noise_sigma:.4f}",
        flush=True,
    )
    logger.info(
        f"Trial {trial.number:03d} | ddpg={ddpg_params} | "
        f"noise_sigma={noise_sigma:.4f} | weights={reward_weights_a}"
    )

    try:
        mean_reward, run_name = run_trial(
            ddpg_params, noise_sigma, reward_weights_a,
            reward_type, trial.number, OPTUNA_TIMESTEPS,
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
    ddpg_keys   = list(DDPG_SEARCH_SPACE.keys())
    reward_keys = list(REWARD_SEARCH_SPACE_A.keys())

    ddpg_params  = {k: best_params[k] for k in ddpg_keys if k in best_params}
    noise_sigma  = best_params.get("noise_sigma", 0.1)
    reward_weights_a = None
    if reward_type == "A":
        reward_weights_a = {k: best_params[k] for k in reward_keys if k in best_params}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"DDPG_opt_best_{reward_type}_{RUN_TAG}_{timestamp}"

    logger.info(f"{'=' * 60}")
    logger.info(f"Retreino final: {run_name}")
    logger.info(f"DDPG params:    {ddpg_params}")
    logger.info(f"Noise sigma:    {noise_sigma:.4f}")
    logger.info(f"Reward weights: {reward_weights_a}")
    logger.info(f"Timesteps:      {TOTAL_TIMESTEPS}")
    logger.info(f"{'=' * 60}")

    train_env = DummyVecEnv([make_env(reward_type, SEED, reward_weights_a)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

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

    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(1),
        sigma=noise_sigma * np.ones(1),
    )

    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=0,
        tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"),
        action_noise=action_noise,
        **ddpg_params,
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
        "run_name":          run_name,
        "reward_type":       reward_type,
        "total_timesteps":   TOTAL_TIMESTEPS,
        "ddpg_params":       ddpg_params,
        "noise_sigma":       noise_sigma,
        "reward_weights_a":  reward_weights_a,
        "seed":              SEED,
        "best_mean_reward":  float(eval_callback.best_mean_reward),
        "total_episodes":    len(reward_logger.episode_rewards),
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
            study_name=f"ddpg_reward{rtype}_{RUN_TAG}",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )

        study.optimize(
            lambda trial: objective(trial, rtype),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        logger.info(f"Estudo concluído — melhor trial: {study.best_trial.number}")
        logger.info(f"Melhor valor:    {study.best_value:.4f}")
        logger.info(f"Melhores params: {study.best_params}")

        # Salva resultados do estudo
        study_path = os.path.join(STUDIES_DIR, f"study_ddpg_reward{rtype}_{RUN_TAG}.json")
        study_data = {
            "best_trial":       study.best_trial.number,
            "best_value":       study.best_value,
            "best_params":      study.best_params,
            "n_trials":         N_TRIALS,
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
