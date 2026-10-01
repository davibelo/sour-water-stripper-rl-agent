import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ID = "04"
INPUT_CSV = Path(__file__).with_name(f"sim_results_cleaned-{DATA_ID}.csv")
OUTPUT_DIR = Path(__file__).parent

INPUTS = ["Qfeed", "Tfeed", "cH2S", "cNH3", "Qreb", "Pcolumn", "Ecolumn"]
INPUT_MINS = np.array([47500, 118.95, 875, 750, 2.0, 5.7, 66.5], dtype=float)
INPUT_MAXS = np.array([52500, 125.05, 3500, 3000, 6.0, 6.3, 73.5], dtype=float)

OUTPUTS = ["Tbottom", "Ttop"]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

NORMALIZATION_FILE = OUTPUT_DIR / f"sim_results_normalization-{DATA_ID}.json"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    missing_inputs = [c for c in INPUTS if c not in df.columns]
    missing_outputs = [c for c in OUTPUTS if c not in df.columns]
    if missing_inputs or missing_outputs:
        raise ValueError(
            f"Missing columns. Inputs: {missing_inputs}. Outputs: {missing_outputs}."
        )

    if len(INPUTS) != len(INPUT_MINS) or len(INPUTS) != len(INPUT_MAXS):
        raise ValueError("INPUTS, INPUT_MINS, and INPUT_MAXS must have same length.")

    input_ranges = INPUT_MAXS - INPUT_MINS
    if np.any(input_ranges == 0):
        raise ValueError("Input ranges contain zero values.")

    output_mins = df[OUTPUTS].min().to_numpy(dtype=float)
    output_maxs = df[OUTPUTS].max().to_numpy(dtype=float)
    output_ranges = output_maxs - output_mins
    if np.any(output_ranges == 0):
        raise ValueError("Output ranges contain zero values.")

    df_scaled = df.copy()
    df_scaled[INPUTS] = (df[INPUTS].to_numpy(dtype=float) - INPUT_MINS) / input_ranges
    df_scaled[OUTPUTS] = (df[OUTPUTS].to_numpy(dtype=float) - output_mins) / output_ranges

    total_frac = TRAIN_FRAC + VAL_FRAC + TEST_FRAC
    if not np.isclose(total_frac, 1.0):
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must equal 1.0")

    n = len(df_scaled)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    df_scaled.iloc[train_idx].to_csv(OUTPUT_DIR / f"sim_results_train-{DATA_ID}.csv", index=False)
    df_scaled.iloc[val_idx].to_csv(OUTPUT_DIR / f"sim_results_val-{DATA_ID}.csv", index=False)
    df_scaled.iloc[test_idx].to_csv(OUTPUT_DIR / f"sim_results_test-{DATA_ID}.csv", index=False)

    normalization = {
        "inputs": INPUTS,
        "input_mins": INPUT_MINS.tolist(),
        "input_maxs": INPUT_MAXS.tolist(),
        "outputs": OUTPUTS,
        "output_mins": output_mins.tolist(),
        "output_maxs": output_maxs.tolist(),
    }

    with NORMALIZATION_FILE.open("w", encoding="utf-8") as f:
        json.dump(normalization, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
