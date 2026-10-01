import pandas as pd
import glob
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "sim_results.csv")

# Check if the combined results file already exists, and delete it if it does
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print(f"{OUTPUT_FILE} already exists and has been deleted.")

# Get list of all CSV files in the specified directory that start with 'simulation_results_'
csv_files = glob.glob(os.path.join(SCRIPT_DIR, "sim_results_*.csv"))

# Initialize an empty list to hold each DataFrame
dataframes = []

# Loop through the list of files and read each one into a DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all DataFrames
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Save the combined DataFrame to the specified directory
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print("All files have been combined and saved to sim_results.csv in the specified folder.")
else:
    print("No files found to combine in the specified folder.")
