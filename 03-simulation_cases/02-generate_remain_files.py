import pandas as pd
import os

base_dir = os.path.dirname(__file__)

# File paths
initial_dir = os.path.join(base_dir, "..", "02-amostragem_dnn")
initial_files = sorted(
    os.path.join(initial_dir, f)
    for f in os.listdir(initial_dir)
    if f.lower().endswith(".csv")
)
results_files = [
    os.path.join(base_dir, f"sim_results_part_{i}.csv")
    for i in range(1, len(initial_files) + 1)
]

# Loop through each initial and results file
for i, (initial_file, results_file) in enumerate(zip(initial_files, results_files), start=1):
    # Load initial and results data
    initial_df = pd.read_csv(initial_file)

    # Handle missing sim_results files by creating empty ones with initial columns plus four placeholder columns
    if not os.path.exists(results_file):
        results_columns = initial_df.columns
        pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

    results_df = pd.read_csv(results_file)

    # Filter out rows that are already simulated (present in results)
    simulated_ids = set(results_df['case_id']) if 'case_id' in results_df.columns else set()
    remaining_df = initial_df[~initial_df['case_id'].isin(simulated_ids)]

    # Filter out rows that previously resulted in errors
    error_file = os.path.join(base_dir, f"error_sim_points_part_{i}.csv")
    if os.path.exists(error_file):
        error_df = pd.read_csv(error_file, usecols=['case_id'])
        error_ids = set(error_df['case_id'])
        before = len(remaining_df)
        remaining_df = remaining_df[~remaining_df['case_id'].isin(error_ids)]
        excluded = before - len(remaining_df)
        if excluded > 0:
            print(f"Excluded {excluded} error point(s) from remain file for part {i}.")

    # Shuffle the rows of the remaining dataframe
    #+remaining_df = remaining_df.sample(frac=1).reset_index(drop=True)

    # Define the output file path
    output_file = os.path.join(base_dir, f"remain_sim_points_part_{i}.csv")

    # Check if the output file already exists and delete it if necessary
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} deleted.")

    # Save shuffled remaining points to a new CSV file
    remaining_df.to_csv(output_file, index=False)
    print(f"Remaining points saved to {output_file}")
