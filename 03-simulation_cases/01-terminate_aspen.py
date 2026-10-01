import psutil
import os
import glob

# Define the process names for Aspen Plus executables
process_names = ["apmain", "AspenPlus"]

# Define the path to the simulation file folder
SIM_FOLDER = "aspen_files"

# Function to forcefully terminate Aspen Plus processes
def terminate_aspen_processes():
    for process_name in process_names:
        for process in psutil.process_iter(['name', 'pid']):
            if process.info['name'] and process_name.lower() in process.info['name'].lower():
                try:
                    process.terminate()  # Attempt to terminate the process
                    process.wait(timeout=5)  # Wait briefly for termination
                    print(f"Terminated process {process.info['name']} (PID: {process.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"Failed to terminate process {process.info['name']} (PID: {process.info['pid']})")

# Function to delete all files in the UTAA_run folder except .bkp files
def delete_non_bkp_files(folder_path):
    # Get all files in the folder except .bkp files
    files_to_delete = glob.glob(os.path.join(folder_path, "*"))
    for file_path in files_to_delete:
        if not file_path.endswith(".bkp"):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

# Run the termination and deletion functions
if __name__ == "__main__":
    terminate_aspen_processes()
    delete_non_bkp_files(SIM_FOLDER)
