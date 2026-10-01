import os
import json
import numpy as np

RUN_TAG = "opt01"
DATA_ID = "04"
MODEL_ID = "02b"

REWARD_TYPES = ["A"]
REWARD_EPSILON_B = 1e-3

# Pesos padrão da recompensa A (centro do espaço de busca; usados como fallback)
REWARD_WEIGHTS_A = {
    "w1": 0.105,
    "w2": 0.9,
    "w3": 0.6,
}

# ── Espaços de busca do Optuna ────────────────────────────────────────────────
# Formato: {"type": "float"|"int"|"categorical",
#            "low": ..., "high": ..., "log": True/False,   # para float/int
#            "choices": [...]}                              # para categorical

PPO_SEARCH_SPACE = {
    "learning_rate": {"type": "float",       "low": 1e-5,  "high": 1e-3,  "log": True},
    "n_steps":       {"type": "categorical", "choices": [512, 1024, 2048, 4096]},
    "batch_size":    {"type": "categorical", "choices": [32, 64, 128, 256]},  # filtrado por n_steps em runtime
    "n_epochs":      {"type": "int",         "low": 4,     "high": 20},
    "gamma":         {"type": "float",       "low": 0.95,  "high": 0.999},
    "gae_lambda":    {"type": "float",       "low": 0.9,   "high": 1.0},
    "clip_range":    {"type": "float",       "low": 0.1,   "high": 0.4},
    "ent_coef":      {"type": "float",       "low": 0.0,   "high": 0.1},
    "vf_coef":       {"type": "float",       "low": 0.3,   "high": 0.9},
    "max_grad_norm": {"type": "float",       "low": 0.3,   "high": 1.0},
}

REWARD_SEARCH_SPACE_A = {
    "w1": {"type": "float", "low": 0.05, "high": 0.30},
    "w2": {"type": "float", "low": 0.50, "high": 1.50},
    "w3": {"type": "float", "low": 0.10, "high": 1.50},
}

# ── Configurações do estudo Optuna ────────────────────────────────────────────
N_TRIALS         = 50         # número de trials na busca
OPTUNA_TIMESTEPS = 150_000    # timesteps por trial (busca rápida)
TOTAL_TIMESTEPS  = 500_000    # timesteps do retreino final com os melhores params

EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 20
EPISODE_STEPS   = 30
SEED            = 42

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURROGATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "05-surrogate", "output_files")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
FIGURES_DIR   = os.path.join(BASE_DIR, "results", "figures")
LOGS_DIR      = os.path.join(BASE_DIR, "results", "logs")
STUDIES_DIR   = os.path.join(BASE_DIR, "results", "studies")

SURROGATE_TBOTTOM = os.path.join(SURROGATE_DIR, f"{MODEL_ID}_Tbottom_best.keras")
SURROGATE_TTOP    = os.path.join(SURROGATE_DIR, f"{MODEL_ID}_Ttop_best.keras")
NORM_JSON = os.path.join(
    os.path.dirname(BASE_DIR), "05-surrogate",
    f"sim_results_normalization-{DATA_ID}.json"
)

with open(NORM_JSON, "r") as _f:
    _norm = json.load(_f)

INPUT_NAMES = _norm["inputs"]
INPUT_MINS  = np.array(_norm["input_mins"])
INPUT_MAXS  = np.array(_norm["input_maxs"])
OUTPUT_NAMES = _norm["outputs"]
OUTPUT_MINS  = np.array(_norm["output_mins"])
OUTPUT_MAXS  = np.array(_norm["output_maxs"])

OBS_NAMES = ["Qfeed", "Tfeed", "Qreb", "Pcolumn", "Tbottom", "Ttop"]

QREB_MIN = INPUT_MINS[INPUT_NAMES.index("Qreb")]
QREB_MAX = INPUT_MAXS[INPUT_NAMES.index("Qreb")]

DELTA_QREB_MAX = 0.3
QREB_INIT_1 = QREB_MIN
QREB_INIT_2 = QREB_MIN + 0.33 * (QREB_MAX - QREB_MIN)
QREB_INIT_3 = QREB_MIN + 0.67 * (QREB_MAX - QREB_MIN)
QREB_INIT_4 = QREB_MAX

QREB_INIT_LOW  = [QREB_INIT_1, QREB_INIT_2]
QREB_INIT_MID  = [QREB_INIT_2, QREB_INIT_3]
QREB_INIT_HIGH = [QREB_INIT_3, QREB_INIT_4]

TBOTTOM_SAFE_MIN = OUTPUT_MINS[OUTPUT_NAMES.index("Tbottom")] - 5.0
TBOTTOM_SAFE_MAX = OUTPUT_MAXS[OUTPUT_NAMES.index("Tbottom")] + 5.0
TTOP_SAFE_MIN    = OUTPUT_MINS[OUTPUT_NAMES.index("Ttop")] - 5.0
TTOP_SAFE_MAX    = OUTPUT_MAXS[OUTPUT_NAMES.index("Ttop")] + 5.0
PENALTY_VIOLATION = -50.0
