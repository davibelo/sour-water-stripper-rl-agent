import os
import json
import numpy as np

RUN_TAG = "04"
DATA_ID = "04"
MODEL_ID = "02b"

REWARD_TYPES = ["A"]
REWARD_WEIGHTS_A = {
    "w1": 0.105,
    "w2": 0.9,
    "w3": 0.6,
}
REWARD_EPSILON_B = 1e-3

PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 20
EPISODE_STEPS = 30
SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURROGATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "05-surrogate", "output_files")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
LOGS_DIR = os.path.join(BASE_DIR, "results", "logs")
SURROGATE_TBOTTOM = os.path.join(SURROGATE_DIR, f"{MODEL_ID}_Tbottom_best.keras")
SURROGATE_TTOP = os.path.join(SURROGATE_DIR, f"{MODEL_ID}_Ttop_best.keras")
NORM_JSON = os.path.join(os.path.dirname(BASE_DIR), "05-surrogate", f"sim_results_normalization-{DATA_ID}.json")

with open(NORM_JSON, "r") as _f:
    _norm = json.load(_f)

INPUT_NAMES = _norm["inputs"]
INPUT_MINS = np.array(_norm["input_mins"])
INPUT_MAXS = np.array(_norm["input_maxs"])

OUTPUT_NAMES = _norm["outputs"]
OUTPUT_MINS = np.array(_norm["output_mins"])
OUTPUT_MAXS = np.array(_norm["output_maxs"])

OBS_NAMES = ["Qfeed", "Tfeed", "Qreb", "Pcolumn", "Tbottom", "Ttop"]

QREB_MIN = INPUT_MINS[INPUT_NAMES.index("Qreb")]
QREB_MAX = INPUT_MAXS[INPUT_NAMES.index("Qreb")]

DELTA_QREB_MAX = 0.3
QREB_INIT_1 = QREB_MIN
QREB_INIT_2 = QREB_MIN + 0.33 * (QREB_MAX - QREB_MIN)
QREB_INIT_3 = QREB_MIN + 0.67 * (QREB_MAX - QREB_MIN)
QREB_INIT_4 = QREB_MAX

QREB_INIT_LOW   = [QREB_INIT_1, QREB_INIT_2]
QREB_INIT_MID   = [QREB_INIT_2, QREB_INIT_3]
QREB_INIT_HIGH  = [QREB_INIT_3, QREB_INIT_4]

TBOTTOM_SAFE_MIN = OUTPUT_MINS[OUTPUT_NAMES.index("Tbottom")] - 5.0
TBOTTOM_SAFE_MAX = OUTPUT_MAXS[OUTPUT_NAMES.index("Tbottom")] + 5.0
TTOP_SAFE_MIN = OUTPUT_MINS[OUTPUT_NAMES.index("Ttop")] - 5.0
TTOP_SAFE_MAX = OUTPUT_MAXS[OUTPUT_NAMES.index("Ttop")] + 5.0
PENALTY_VIOLATION = -50.0