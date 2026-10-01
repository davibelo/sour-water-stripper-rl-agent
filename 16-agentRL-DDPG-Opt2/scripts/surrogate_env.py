from collections import deque
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    DELTA_QREB_MAX,
    EPISODE_STEPS,
    INPUT_MAXS,
    INPUT_MINS,
    MODEL_ID,
    OUTPUT_MAXS,
    OUTPUT_MINS,
    PENALTY_VIOLATION,
    QREB_INIT_HIGH,
    QREB_INIT_LOW,
    QREB_INIT_MID,
    QREB_MAX,
    QREB_MIN,
    REWARD_EPSILON_B,
    REWARD_WEIGHTS_A,
    SURROGATE_DIR,
    TBOTTOM_SAFE_MAX,
    TBOTTOM_SAFE_MIN,
    TTOP_SAFE_MAX,
    TTOP_SAFE_MIN,
)


def _load_surrogates():
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    single_path  = os.path.join(SURROGATE_DIR, f"{MODEL_ID}_best.keras")
    tbottom_path = os.path.join(SURROGATE_DIR, f"{MODEL_ID}_Tbottom_best.keras")
    ttop_path    = os.path.join(SURROGATE_DIR, f"{MODEL_ID}_Ttop_best.keras")

    if os.path.exists(single_path):
        model_single = tf.keras.models.load_model(single_path, compile=False)
        return "single", model_single, None, None
    elif os.path.exists(tbottom_path) and os.path.exists(ttop_path):
        model_tbottom = tf.keras.models.load_model(tbottom_path, compile=False)
        model_ttop    = tf.keras.models.load_model(ttop_path, compile=False)
        return "dual", None, model_tbottom, model_ttop
    else:
        raise FileNotFoundError(
            f"No surrogate model found for MODEL_ID='{MODEL_ID}' in '{SURROGATE_DIR}'. "
            f"Expected '{single_path}' or both '{tbottom_path}' and '{ttop_path}'."
        )


class StripperSurrogateEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_type="A", seed=None, reward_weights_a=None):
        """
        reward_weights_a: dict com chaves w1, w2, w3 para a recompensa tipo A.
            Se None, usa REWARD_WEIGHTS_A do config (valores padrão).
            Permite que o Optuna passe pesos diferentes por trial.
        """
        super().__init__()
        assert reward_type in ("A", "B"), "reward_type must be 'A' or 'B'"
        self.reward_type = reward_type
        self._reward_weights_a = reward_weights_a if reward_weights_a is not None else REWARD_WEIGHTS_A

        self.surrogate_type, self.model_single, self.model_tbottom, self.model_ttop = _load_surrogates()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        self.input_mins   = INPUT_MINS.copy()
        self.input_maxs   = INPUT_MAXS.copy()
        self.input_ranges = self.input_maxs - self.input_mins
        self.output_mins   = OUTPUT_MINS.copy()
        self.output_maxs   = OUTPUT_MAXS.copy()
        self.output_ranges = self.output_maxs - self.output_mins

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._tbottom_history = deque(maxlen=5)

        self.Qfeed   = 0.0
        self.Tfeed   = 0.0
        self.cH2S    = 0.0
        self.cNH3    = 0.0
        self.Qreb    = 0.0
        self.Pcolumn = 0.0
        self.Ecolumn = 0.0
        self.Tbottom = 0.0
        self.Ttop    = 0.0

    def _normalize_input(self, raw_vec):
        return (raw_vec - self.input_mins) / self.input_ranges

    def _denormalize_output(self, scaled_val, idx):
        return scaled_val * self.output_ranges[idx] + self.output_mins[idx]

    def _predict(self):
        raw    = np.array([self.Qfeed, self.Tfeed, self.cH2S, self.cNH3,
                           self.Qreb, self.Pcolumn, self.Ecolumn])
        x_norm = self._normalize_input(raw).reshape(1, -1).astype(np.float32)
        if self.surrogate_type == "single":
            out      = self.model_single(x_norm, training=False).numpy().flatten()
            tb_scaled = float(out[0])
            tt_scaled = float(out[1])
        else:
            tb_scaled = float(self.model_tbottom(x_norm, training=False).numpy().flatten()[0])
            tt_scaled = float(self.model_ttop(x_norm,    training=False).numpy().flatten()[0])
        self.Tbottom = self._denormalize_output(tb_scaled, 0)
        self.Ttop    = self._denormalize_output(tt_scaled, 1)

    def _get_obs(self):
        obs_raw  = np.array([self.Qfeed, self.Tfeed, self.Qreb, self.Pcolumn,
                              self.Tbottom, self.Ttop])
        obs_mins = np.array([self.input_mins[0], self.input_mins[1], self.input_mins[4],
                              self.input_mins[5], self.output_mins[0], self.output_mins[1]])
        obs_maxs = np.array([self.input_maxs[0], self.input_maxs[1], self.input_maxs[4],
                              self.input_maxs[5], self.output_maxs[0], self.output_maxs[1]])
        obs_norm = (obs_raw - obs_mins) / (obs_maxs - obs_mins)
        return np.clip(obs_norm, 0.0, 1.0).astype(np.float32)

    def _check_safety(self):
        if self.Tbottom < TBOTTOM_SAFE_MIN or self.Tbottom > TBOTTOM_SAFE_MAX:
            return True
        if self.Ttop < TTOP_SAFE_MIN or self.Ttop > TTOP_SAFE_MAX:
            return True
        return False

    def _compute_reward(self):
        if self._check_safety():
            return PENALTY_VIOLATION, True

        if self.reward_type == "A":
            w      = self._reward_weights_a
            var_tb = np.var(list(self._tbottom_history)) if len(self._tbottom_history) >= 2 else 0.0
            reward = (
                w["w1"] * (self.Tbottom - self.Ttop)
                - w["w2"] * self.Qreb
                - w["w3"] * var_tb
            )
        else:
            reward = (self.Tbottom - self.Ttop) / (self.Qreb + REWARD_EPSILON_B)

        return float(reward), False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.Qfeed   = self._rng.uniform(self.input_mins[0], self.input_maxs[0])
        self.Tfeed   = self._rng.uniform(self.input_mins[1], self.input_maxs[1])
        self.cH2S    = self._rng.uniform(self.input_mins[2], self.input_maxs[2])
        self.cNH3    = self._rng.uniform(self.input_mins[3], self.input_maxs[3])
        self.Pcolumn = self._rng.uniform(self.input_mins[5], self.input_maxs[5])
        self.Ecolumn = self._rng.uniform(self.input_mins[6], self.input_maxs[6])

        init_mode = self._rng.choice(["low", "mid", "high"])
        if init_mode == "low":
            self.Qreb = self._rng.uniform(QREB_INIT_LOW[0],  QREB_INIT_LOW[1])
        elif init_mode == "mid":
            self.Qreb = self._rng.uniform(QREB_INIT_MID[0],  QREB_INIT_MID[1])
        else:
            self.Qreb = self._rng.uniform(QREB_INIT_HIGH[0], QREB_INIT_HIGH[1])

        self._predict()
        self._step_count = 0
        self._tbottom_history.clear()
        self._tbottom_history.append(self.Tbottom)

        return self._get_obs(), {}

    def step(self, action):
        delta_qreb = float(action[0]) * DELTA_QREB_MAX
        self.Qreb  = np.clip(self.Qreb + delta_qreb, QREB_MIN, QREB_MAX)

        self._predict()
        self._tbottom_history.append(self.Tbottom)
        self._step_count += 1

        reward, violated = self._compute_reward()
        terminated = violated
        truncated  = self._step_count >= EPISODE_STEPS

        info = {
            "Qreb":    self.Qreb,
            "Tbottom": self.Tbottom,
            "Ttop":    self.Ttop,
            "delta_T": self.Tbottom - self.Ttop,
            "step":    self._step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info
