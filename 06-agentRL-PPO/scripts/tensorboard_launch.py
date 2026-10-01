import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
VENV_PYTHON = os.path.join(WORKSPACE_DIR, ".venv", "Scripts", "python.exe")
LOGDIR = os.path.join(BASE_DIR, "results", "logs", "tensorboard")


def main():
    python_exe = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
    code = f"import sys; sys.argv = ['tensorboard', '--logdir', r'{LOGDIR}']; from tensorboard import main; main.run_main()"
    subprocess.run([python_exe, "-c", code], check=True)


if __name__ == "__main__":
    main()
