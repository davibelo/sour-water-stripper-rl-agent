import os
import shutil
import csv
import concurrent.futures
from multiprocessing import Manager, freeze_support, Process
import win32com.client as win32
import time
import pythoncom

INPUT_FILE_TEMPLATE = 'remain_sim_points_part_{}.csv'
RESULT_FILE_TEMPLATE = 'sim_results_part_{}.csv'
ERROR_FILE_TEMPLATE = 'error_sim_points_part_{}.csv'  # Error file template
SIMULATION_FILE = r'aspen_files\Simulacao_revF.bkp'
NUM_INSTANCES = 2
INIT_DELAY_SECONDS = 5

base_dir = os.path.dirname(__file__)
aspen_Path = os.path.abspath(os.path.join(base_dir, SIMULATION_FILE))

def log_worker(log_queue, log_file_path):
    """Worker that listens to log_queue and writes error messages to the log file."""
    buffer = []
    flush_interval = 1  # seconds
    last_flush_time = time.time()

    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        while True:
            try:
                # Timeout allows us to periodically flush the buffer
                message = log_queue.get(timeout=flush_interval)
                if message == "STOP":
                    break
                buffer.append(message)
                print(message)  # Limited console output for monitoring
            except:
                pass  # Timeout reached; proceed to flush buffer

            # Periodic buffer flush
            current_time = time.time()
            if current_time - last_flush_time >= flush_interval or len(buffer) > 100:
                if buffer:
                    log_file.write('\n'.join(buffer) + '\n')
                    log_file.flush()
                    buffer.clear()
                    last_flush_time = current_time

def log_message_factory(log_queue):
    """Creates a logging function that adds only error messages to the log queue."""
    def log_message(message, is_error=False):
        if is_error:
            log_queue.put(message)
        else:
            print(message)  # Print successful runs to the console only
    return log_message

def load_points_from_csv(filename):
    points = []
    fieldnames = []
    with open(filename, mode='r', encoding='utf-8', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            try:
                point = (
                    int(row['case_id'].strip()),
                    float(row['Qfeed'].strip()),
                    float(row['Tfeed'].strip()),
                    float(row['cH2S'].strip()),
                    float(row['cNH3'].strip()),
                    float(row['Qreb'].strip()),
                    float(row['Pcolumn'].strip()),
                    float(row['Ecolumn'].strip()),
                )
            except (KeyError, AttributeError, ValueError):
                continue
            points.append(point)
    return points, fieldnames

def save_result_to_csv(result, result_file):
    """Appends a single result row to the specified results CSV file."""
    with open(result_file, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(result)

def save_error_to_csv(point, error_file):
    """Logs an error point to the specified error CSV file."""
    with open(error_file, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list(point) + ['', ''])

def start_aspen(instance_id, log_message):
    """Starts Aspen using the instance-specific copy of the simulation file and returns the Application object."""
    instance_file = f'{aspen_Path}_copy_{instance_id}.bkp'
    shutil.copyfile(aspen_Path, instance_file)

    try:
        Application = win32.Dispatch('Apwn.Document')
        Application.InitFromArchive2(instance_file)
        Application.visible = 0
        log_message(f"Aspen started successfully for instance {instance_id}.")
        return Application
    except Exception as e:
        log_message(f"Failed to start Aspen for instance {instance_id}: {e}", is_error=True)
        return None

def warmup_run(Application, log_message):
    try:
        Application.Engine.Run2()
        test_paths = [
            r"\Data\Streams\8\Input\TOTFLOW\MIXED",
            r"\Data\Blocks\T1\Input\STEFF_SEC\1",
            r"\Data\Streams\14\Output\TEMP_OUT\MIXED",
            r"\Data\Streams\11\Output\TEMP_OUT\MIXED",
        ]
        for path in test_paths:
            node = Application.Tree.FindNode(path)
            if node is None:
                log_message(f"Warmup: node still missing after Run2: {path}", is_error=True)
                return False
        log_message("Warmup run completed, all nodes accessible.")
        return True
    except Exception as e:
        log_message(f"Warmup run failed: {e}", is_error=True)
        return False

def simulate(x, Application, log_message, lock, result_file, error_file):
    case_id, Qfeed, Tfeed, cH2S, cNH3, Qreb, Pcolumn, Ecolumn = x

    if not Application:
        Application = start_aspen(log_message)
        if not Application:
            log_message(f"Failed to initialize Aspen for inputs {x}. Skipping simulation.", is_error=True)
            return None

    cH2O = 1-(cH2S/1E6)-(cNH3/1E6)

    try:
        inputs = [
            (r"\Data\Streams\8\Input\TOTFLOW\MIXED", Qfeed),
            (r"\Data\Streams\8\Input\TEMP\MIXED", Tfeed),
            (r"\Data\Streams\8\Input\FLOW\MIXED\H2S", cH2S/1E6),
            (r"\Data\Streams\8\Input\FLOW\MIXED\NH3", cNH3/1E6),
            (r"\Data\Streams\8\Input\FLOW\MIXED\H2O", cH2O),
            (r"\Data\Blocks\T1\Input\QN", Qreb),
            (r"\Data\Blocks\T1\Input\PRES1", Pcolumn),
            (r"\Data\Blocks\T1\Input\STEFF_SEC\1", Ecolumn/100),
        ]
        for path, value in inputs:
            node = Application.Tree.FindNode(path)
            if node is None:
                raise ValueError(f"Node not found: {path}")
            node.Value = value

        Application.Engine.Run2()

        per_error_node = Application.Tree.FindNode(
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
            convergence_msg = " ".join(error_lines)
            log_message(
                f"Convergence error for {x}: {convergence_msg}",
                is_error=True
            )
            with lock:
                save_error_to_csv(x, error_file)
            return None

        outputs = [
            r"\Data\Streams\14\Output\TEMP_OUT\MIXED",
            r"\Data\Streams\11\Output\TEMP_OUT\MIXED",
        ]
        y = []
        for path in outputs:
            node = Application.Tree.FindNode(path)
            if node is None:
                raise ValueError(f"Node not found: {path}")
            y.append(node.Value)
        y = tuple(y)

        log_message(f"Simulation result: {x} -> Tbottom: {y[0]}, Ttop: {y[1]}")
        with lock:
            save_result_to_csv(x + y, result_file)

        return x + y

    except Exception as e:
        error_message = str(e)
        log_message(f"Error simulating {x}: {error_message}", is_error=True)
        if 'NoneType' in error_message:
            with lock:
                save_error_to_csv(x, error_file)
        return None

def run_parallel_simulations(batch_id, input_file, log_queue, lock):
    pythoncom.CoInitialize()
    log_message = log_message_factory(log_queue)
    result_file = os.path.join(base_dir, RESULT_FILE_TEMPLATE.format(batch_id))
    error_file = os.path.join(base_dir, ERROR_FILE_TEMPLATE.format(batch_id))

    points, fieldnames = load_points_from_csv(input_file)

    if not os.path.exists(result_file):
        with open(result_file, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(fieldnames + ['Tbottom', 'Ttop'])
    if not os.path.exists(error_file):
        with open(error_file, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(fieldnames)

    Application = start_aspen(batch_id, log_message)
    if Application:
        warmup_run(Application, log_message)
    results = []

    for point in points:
        result = simulate(point, Application, log_message, lock, result_file, error_file)
        if result:
            results.append(result)

    if Application:
        try:
            Application.Close()
        except Exception as e:
            log_message(f"Error closing Aspen for instance {batch_id}: {e}", is_error=True)
    pythoncom.CoUninitialize()
    return results

if __name__ == '__main__':
    freeze_support()

    # Define the path to the log file
    log_file_path = f"{os.path.splitext(__file__)[0]}.log"

    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Create a manager and lock for synchronizing file access
    manager = Manager()
    lock = manager.Lock()

    # Set up a global logging queue and process
    log_queue = manager.Queue()
    log_process = Process(target=log_worker, args=(log_queue, log_file_path))
    log_process.start()

    # Run simulations in parallel for each input file
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_INSTANCES) as executor:
        futures = [
            executor.submit(
                run_parallel_simulations,
                i + 1,
                os.path.join(base_dir, INPUT_FILE_TEMPLATE.format(i + 1)),
                log_queue,
                lock,
            )
            for i in range(NUM_INSTANCES)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker failed with error: {e}")

    # Stop the logging process
    log_queue.put("STOP")
    log_process.join()

    print('Simulation results saved in multiple result files.')
