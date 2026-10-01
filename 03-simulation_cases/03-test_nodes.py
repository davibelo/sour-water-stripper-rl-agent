import os
import win32com.client as win32

SIMULATION_FILE = r'aspen_files\Simulacao_revE.bkp'

base_dir = os.path.dirname(__file__)
aspen_path = os.path.abspath(os.path.join(base_dir, SIMULATION_FILE))

paths = [
    r"\Data\Streams\8\Input\TOTFLOW\MIXED",
    r"\Data\Streams\8\Input\TEMP\MIXED",
    r"\Data\Streams\8\Input\FLOW\MIXED\H2S",
    r"\Data\Streams\8\Input\FLOW\MIXED\NH3",
    r"\Data\Streams\8\Input\FLOW\MIXED\H2O",
    r"\Data\Blocks\T1\Input\QN",
    r"\Data\Blocks\T1\Input\PRES1",
    r"\Data\Blocks\T1\Input\STEFF_SEC\1"
]

Application = win32.Dispatch('Apwn.Document')
Application.InitFromArchive2(aspen_path)
Application.visible = 0

for path in paths:
    node = Application.Tree.FindNode(path)
    if node is None:
        print(f"MISSING|{path}")
    else:
        try:
            value = node.Value
        except Exception as e:
            value = f"ERROR: {e}"
        print(f"FOUND|{path}|{value}")

Application.Close()
