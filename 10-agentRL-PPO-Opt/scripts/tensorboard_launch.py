"""Lança o TensorBoard contornando problemas de PATH no Windows com uv."""
import sys
import os

logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "logs", "tensorboard")

sys.argv = ["tensorboard", f"--logdir={logs_dir}", "--port=6006"]

from tensorboard import main as tb_main
tb_main.run_main()
