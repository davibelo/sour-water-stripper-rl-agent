import os
import csv
import shutil
import time
import pythoncom
import win32com.client as win32

# ── Adjustable parameters ─────────────────────────────────────────────────────
RUN    = "opt02"   # deve corresponder ao RUN usado em 01_simulate_agent.py
REWARD = "A"
BANDS  = ["low", "mid", "high"]
SIMULATION_FILE  = r"03-simulation_cases\aspen_files\Simulacao_revF.bkp"
INIT_DELAY_SECONDS = 5
# ─────────────────────────────────────────────────────────────────────────────

base_dir          = os.path.dirname(os.path.abspath(__file__))
data_dir          = os.path.join(base_dir, "data")
final_results_dir = os.path.join(base_dir, "final_results")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(final_results_dir, exist_ok=True)
aspen_path = os.path.abspath(os.path.join(base_dir, "..", SIMULATION_FILE))

EPISODES_RESULTS_FILE = os.path.join(data_dir, f"episodes_results_{RUN}_{REWARD}.csv")
FINAL_RESULTS_FILE    = os.path.join(final_results_dir, f"sim_results_{RUN}_{REWARD}.csv")

SIM_POINTS_TEMPLATE  = os.path.join(data_dir, f"sim_points_{RUN}_{REWARD}_{{band}}.csv")
SIM_RESULTS_TEMPLATE = os.path.join(data_dir, f"sim_results_{RUN}_{REWARD}_{{band}}.csv")
ERROR_TEMPLATE       = os.path.join(data_dir, f"error_sim_points_{RUN}_{REWARD}_{{band}}.csv")

INPUT_PATHS = [
    (r"\Data\Streams\8\Input\TOTFLOW\MIXED",   "Qfeed"),
    (r"\Data\Streams\8\Input\TEMP\MIXED",       "Tfeed"),
    (r"\Data\Streams\8\Input\FLOW\MIXED\H2S",   "cH2S_frac"),
    (r"\Data\Streams\8\Input\FLOW\MIXED\NH3",   "cNH3_frac"),
    (r"\Data\Streams\8\Input\FLOW\MIXED\H2O",   "cH2O"),
    (r"\Data\Blocks\T1\Input\QN",               "Qreb"),
    (r"\Data\Blocks\T1\Input\PRES1",            "Pcolumn"),
    (r"\Data\Blocks\T1\Input\STEFF_SEC\1",      "Ecolumn_frac"),
]

OUTPUT_PATHS = {
    "Tbottom":    r"\Data\Streams\14\Output\TEMP_OUT\MIXED",
    "Ttop":       r"\Data\Streams\11\Output\TEMP_OUT\MIXED",
    "H2S_Recover": r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\5",
    "NH3_Loss":    r"\Data\Flowsheeting Options\Calculator\C-1\Output\WRITE_VAL\6",
}

RESULT_FIELDNAMES = [
    "case_id", "Qfeed", "Tfeed", "cH2S", "cNH3",
    "Qreb", "Pcolumn", "Ecolumn",
    "Tbottom", "Ttop", "H2S_Recover", "NH3_Loss",
    "Qreb_final", "n_iter_final",
]


def load_episodes_lookup(episodes_file):
    lookup = {}
    with open(episodes_file, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[int(row["episode_id"])] = int(row["n_iter_final"])
    return lookup


def load_sim_points(sim_file):
    points = []
    with open(sim_file, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append({
                "case_id":    int(row["case_id"]),
                "Qfeed":      float(row["Qfeed"]),
                "Tfeed":      float(row["Tfeed"]),
                "cH2S":       float(row["cH2S"]),
                "cNH3":       float(row["cNH3"]),
                "Qreb":       float(row["Qreb"]),
                "Pcolumn":    float(row["Pcolumn"]),
                "Ecolumn":    float(row["Ecolumn"]),
                "Qreb_final": float(row["Qreb_final"]),
            })
    return points


def start_aspen(instance_id):
    instance_file = f"{aspen_path}_copy_{instance_id}.bkp"
    shutil.copyfile(aspen_path, instance_file)
    try:
        app = win32.Dispatch("Apwn.Document")
        app.InitFromArchive2(instance_file)
        app.visible = 0
        print(f"Aspen started for instance {instance_id}.")
        return app
    except Exception as e:
        print(f"ERROR: Failed to start Aspen instance {instance_id}: {e}")
        return None


def warmup_run(app):
    try:
        app.Engine.Run2()
        test_paths = [
            r"\Data\Streams\8\Input\TOTFLOW\MIXED",
            r"\Data\Blocks\T1\Input\STEFF_SEC\1",
            r"\Data\Streams\14\Output\TEMP_OUT\MIXED",
            r"\Data\Streams\11\Output\TEMP_OUT\MIXED",
        ]
        for path in test_paths:
            node = app.Tree.FindNode(path)
            if node is None:
                print(f"WARNING: Warmup node missing: {path}")
                return False
        print("Warmup run completed.")
        return True
    except Exception as e:
        print(f"ERROR: Warmup failed: {e}")
        return False


def simulate_point(point, app, result_file, error_file, n_iter_final):
    case_id    = point["case_id"]
    Qfeed      = point["Qfeed"]
    Tfeed      = point["Tfeed"]
    cH2S       = point["cH2S"]
    cNH3       = point["cNH3"]
    Qreb       = point["Qreb"]
    Pcolumn    = point["Pcolumn"]
    Ecolumn    = point["Ecolumn"]
    Qreb_final = point["Qreb_final"]

    cH2S_frac = cH2S / 1e6
    cNH3_frac = cNH3 / 1e6
    cH2O      = 1.0 - cH2S_frac - cNH3_frac

    values_by_name = {
        "Qfeed":        Qfeed,
        "Tfeed":        Tfeed,
        "cH2S_frac":    cH2S_frac,
        "cNH3_frac":    cNH3_frac,
        "cH2O":         cH2O,
        "Qreb":         Qreb_final,
        "Pcolumn":      Pcolumn,
        "Ecolumn_frac": Ecolumn / 100,
    }

    try:
        inputs = [(path, values_by_name[name]) for path, name in INPUT_PATHS]
        for path, value in inputs:
            node = app.Tree.FindNode(path)
            if node is None:
                raise ValueError(f"Node not found: {path}")
            node.Value = value

        app.Engine.Run2()

        per_error_node = app.Tree.FindNode(
            r"\Data\Results Summary\Run-Status\Output\PER_ERROR"
        )
        if per_error_node is not None and per_error_node.Elements.Count > 0:
            error_lines = []
            for i in range(1, per_error_node.Elements.Count + 1):
                try:
                    err_node = per_error_node.Elements.Item(i)
                    try:
                        msg = err_node.Value
                    except Exception:
                        msg = err_node.Name
                    error_lines.append(str(msg))
                except Exception:
                    pass
            print(f"ERROR convergence case_id={case_id}: {' '.join(error_lines)}")
            with open(error_file, mode="a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    case_id, Qfeed, Tfeed, cH2S, cNH3,
                    Qreb, Pcolumn, Ecolumn,
                ])
            return None

        outputs = {}
        for key, path in OUTPUT_PATHS.items():
            node = app.Tree.FindNode(path)
            if node is None:
                raise ValueError(f"Output node not found: {path}")
            outputs[key] = node.Value

        row = {
            "case_id":     case_id,
            "Qfeed":       Qfeed,
            "Tfeed":       Tfeed,
            "cH2S":        cH2S,
            "cNH3":        cNH3,
            "Qreb":        Qreb,
            "Pcolumn":     Pcolumn,
            "Ecolumn":     Ecolumn,
            "Tbottom":     outputs["Tbottom"],
            "Ttop":        outputs["Ttop"],
            "H2S_Recover": outputs["H2S_Recover"],
            "NH3_Loss":    outputs["NH3_Loss"],
            "Qreb_final":  Qreb_final,
            "n_iter_final": n_iter_final,
        }
        with open(result_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
            writer.writerow(row)

        print(
            f"  case_id={case_id} | Tbottom={outputs['Tbottom']:.2f} | "
            f"Ttop={outputs['Ttop']:.2f} | H2S_Recover={outputs['H2S_Recover']:.4f} | "
            f"NH3_Loss={outputs['NH3_Loss']:.4f}"
        )
        return row

    except Exception as e:
        print(f"ERROR simulating case_id={case_id}: {e}")
        with open(error_file, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                case_id, Qfeed, Tfeed, cH2S, cNH3,
                Qreb, Pcolumn, Ecolumn,
            ])
        return None


def simulate_band(band, episodes_lookup):
    sim_file    = SIM_POINTS_TEMPLATE.format(band=band)
    result_file = SIM_RESULTS_TEMPLATE.format(band=band)
    error_file  = ERROR_TEMPLATE.format(band=band)

    if not os.path.exists(sim_file):
        print(f"WARNING: Sim points file not found, skipping band '{band}': {sim_file}")
        return []

    points = load_sim_points(sim_file)
    print(f"\nSimulating band '{band}' ({len(points)} points)...")

    with open(result_file, mode="w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES).writeheader()
    with open(error_file, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "case_id", "Qfeed", "Tfeed", "cH2S", "cNH3",
            "Qreb", "Pcolumn", "Ecolumn",
        ])

    pythoncom.CoInitialize()
    app = start_aspen(band)
    if app is None:
        pythoncom.CoUninitialize()
        return []

    time.sleep(INIT_DELAY_SECONDS)
    warmup_run(app)

    band_results = []
    for point in points:
        n_iter = episodes_lookup.get(point["case_id"], None)
        result = simulate_point(point, app, result_file, error_file, n_iter)
        if result is not None:
            band_results.append(result)

    try:
        app.Close()
    except Exception as e:
        print(f"WARNING: Error closing Aspen for band '{band}': {e}")

    pythoncom.CoUninitialize()
    return band_results


def merge_results():
    all_rows = []
    for band in BANDS:
        result_file = SIM_RESULTS_TEMPLATE.format(band=band)
        if not os.path.exists(result_file):
            print(f"WARNING: Result file not found, skipping band '{band}': {result_file}")
            continue
        with open(result_file, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["band"] = band
                all_rows.append(row)

    if not all_rows:
        print("No results to merge.")
        return

    final_fieldnames = ["band"] + RESULT_FIELDNAMES
    with open(FINAL_RESULTS_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nFinal merged results saved to: {FINAL_RESULTS_FILE} ({len(all_rows)} rows)")


def main():
    if not os.path.exists(EPISODES_RESULTS_FILE):
        raise FileNotFoundError(
            f"Episodes results file not found: {EPISODES_RESULTS_FILE}\n"
            f"Run 01_simulate_agent.py first."
        )

    episodes_lookup = load_episodes_lookup(EPISODES_RESULTS_FILE)
    print(f"Loaded n_iter_final for {len(episodes_lookup)} episodes.")

    for band in BANDS:
        simulate_band(band, episodes_lookup)

    merge_results()


if __name__ == "__main__":
    main()
