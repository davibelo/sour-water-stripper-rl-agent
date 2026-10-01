import os
import sys
import json
import logging
from collections import deque
from datetime import datetime

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from config import (
    ACTION_NOISE_SIGMA,
    DDPG_HYPERPARAMS,
    EVAL_FREQ,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    N_EVAL_EPISODES,
    REWARD_TYPES,
    RUN_TAG,
    SEED,
    TOTAL_TIMESTEPS,
)
from surrogate_env import StripperSurrogateEnv

LOG_FILE = os.path.join(LOGS_DIR, f"train_ddpg-{RUN_TAG}.log" if RUN_TAG else "train_ddpg.log")


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


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_interval=5000, n_traj_store=5, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.n_traj_store = n_traj_store
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = None
        self.episode_log = []
        self._ep_qreb = None
        self._ep_dt = None
        self._ep_rew = None
        self._ep_first_qreb = None
        self._late_trajs = None

    def _on_training_start(self):
        n = self.training_env.num_envs
        self.current_rewards = np.zeros(n)
        self._ep_qreb = [[] for _ in range(n)]
        self._ep_dt = [[] for _ in range(n)]
        self._ep_rew = [[] for _ in range(n)]
        self._ep_first_qreb = [None] * n
        self._late_trajs = deque(maxlen=self.n_traj_store)

    def _on_step(self):
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        self.current_rewards += rewards

        for i, (done, info, rew) in enumerate(zip(dones, infos, rewards)):
            qreb = info.get("Qreb")
            dt = info.get("delta_T")
            if qreb is not None:
                if self._ep_first_qreb[i] is None:
                    self._ep_first_qreb[i] = qreb
                self._ep_qreb[i].append(qreb)
            if dt is not None:
                self._ep_dt[i].append(dt)
            self._ep_rew[i].append(float(rew))

            if done:
                ep_num = len(self.episode_log)
                ql = self._ep_qreb[i]
                dl = self._ep_dt[i]
                rl = self._ep_rew[i]
                traj = {"Qreb": list(ql), "delta_T": list(dl), "rewards": list(rl)}
                self._late_trajs.append({"episode": ep_num, **traj})
                ep_entry = {
                    "episode": ep_num,
                    "Qreb_init": self._ep_first_qreb[i],
                    "Qreb_final": ql[-1] if ql else None,
                    "Qreb_mean": float(np.mean(ql)) if ql else None,
                    "var_Qreb": float(np.var(ql)) if ql else None,
                    "deltaT_final": dl[-1] if dl else None,
                    "reward_mean": float(np.mean(rl)) if rl else None,
                    "reward_sum": float(self.current_rewards[i]),
                }
                if ep_num < self.n_traj_store:
                    ep_entry["trajectory"] = traj
                self.episode_log.append(ep_entry)
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(info.get("step", 0))
                self.current_rewards[i] = 0.0
                self._ep_qreb[i] = []
                self._ep_dt[i] = []
                self._ep_rew[i] = []
                self._ep_first_qreb[i] = None

        if self.num_timesteps % self.log_interval < self.training_env.num_envs:
            if len(self.episode_rewards) > 0:
                recent = self.episode_rewards[-50:]
                logger.info(
                    f"Step {self.num_timesteps:>8d} | "
                    f"Episodes: {len(self.episode_rewards):>5d} | "
                    f"Mean reward (last 50): {np.mean(recent):>8.3f} | "
                    f"Std: {np.std(recent):>7.3f}"
                )
        return True


def make_env(reward_type, seed):
    def _init():
        env = StripperSurrogateEnv(reward_type=reward_type, seed=seed)
        env = Monitor(env)
        return env
    return _init


def train(reward_type, total_timesteps, tag=""):
    run_name = f"DDPG_reward{reward_type}"
    if tag:
        run_name += f"_{tag}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name += f"_{timestamp}"

    logger.info(f"={'=' * 60}")
    logger.info(f"Starting training: {run_name}")
    logger.info(f"Reward type: {reward_type}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"DDPG hyperparams: {DDPG_HYPERPARAMS}")
    logger.info(f"Action noise sigma: {ACTION_NOISE_SIGMA}")
    logger.info(f"={'=' * 60}")

    train_env = DummyVecEnv([make_env(reward_type, SEED)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(reward_type, SEED + 1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

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
        sigma=ACTION_NOISE_SIGMA * np.ones(1),
    )

    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=0,
        tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"),
        action_noise=action_noise,
        **DDPG_HYPERPARAMS,
    )

    logger.info(f"Policy architecture: {model.policy}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, reward_logger],
        progress_bar=True,
    )

    final_path = os.path.join(model_save_path, "final_model")
    model.save(final_path)
    train_env.save(os.path.join(model_save_path, "vec_normalize.pkl"))

    logger.info(f"Final model saved to {final_path}")
    logger.info(f"VecNormalize stats saved to {model_save_path}/vec_normalize.pkl")

    meta = {
        "run_name": run_name,
        "reward_type": reward_type,
        "total_timesteps": total_timesteps,
        "ddpg_hyperparams": DDPG_HYPERPARAMS,
        "action_noise_sigma": ACTION_NOISE_SIGMA,
        "seed": SEED,
        "episode_rewards_mean": float(np.mean(reward_logger.episode_rewards[-100:])) if reward_logger.episode_rewards else 0.0,
        "episode_rewards_std": float(np.std(reward_logger.episode_rewards[-100:])) if reward_logger.episode_rewards else 0.0,
        "total_episodes": len(reward_logger.episode_rewards),
    }
    with open(os.path.join(model_save_path, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for traj in reward_logger._late_trajs:
        ep_idx = traj["episode"]
        if ep_idx < len(reward_logger.episode_log):
            reward_logger.episode_log[ep_idx]["trajectory"] = {
                k: v for k, v in traj.items() if k != "episode"
            }
    ep_log_path = os.path.join(model_save_path, "episode_log.json")
    with open(ep_log_path, "w") as f:
        json.dump(reward_logger.episode_log, f)
    logger.info(f"Episode log saved to {ep_log_path} ({len(reward_logger.episode_log)} episodes)")

    logger.info(f"Training complete: {run_name}")
    logger.info(f"Total episodes: {meta['total_episodes']}")
    logger.info(f"Mean reward (last 100): {meta['episode_rewards_mean']:.4f}")

    return model, train_env, reward_logger


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for rtype in REWARD_TYPES:
        train(reward_type=rtype, total_timesteps=TOTAL_TIMESTEPS, tag=RUN_TAG)


if __name__ == "__main__":
    main()
