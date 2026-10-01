import subprocess
import time
import psutil
import os

# Path to Python interpreter in the virtual environment
base_dir = os.path.dirname(os.path.realpath(__file__))
python_path = os.path.normpath(os.path.join(base_dir, "..", ".venv", "Scripts", "python.exe"))

def run_script(script_name):
    script_path = os.path.join(base_dir, script_name)
    print(f"  -> {script_path}")
    return subprocess.Popen([python_path, script_path])

def terminate_process_and_children(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        parent.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass
    except subprocess.TimeoutExpired:
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass

while True:
    print("Running 01-terminate_aspen.py")
    run_script("01-terminate_aspen.py").wait()

    time.sleep(5)  # Brief pause to ensure processes are fully terminated

    print("Running 02-generate_remain_files.py")
    run_script("02-generate_remain_files.py").wait()

    print("Running 03-simulate.py")
    simulate_process = run_script("03-simulate.py")

    time.sleep(3600)

    print("Stopping 03-simulate.py and its subprocesses")
    terminate_process_and_children(simulate_process)

    print("Restarting sequence...\n")
    time.sleep(5)
