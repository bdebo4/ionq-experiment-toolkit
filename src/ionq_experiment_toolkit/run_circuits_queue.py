# run_circuits.py
import json, csv, sys, time
from pathlib import Path
from datetime import datetime
import pandas as pd

from qiskit.qpy import load
from qiskit_ionq import IonQProvider
from qiskit import QuantumCircuit,transpile
from qiskit_ionq import ErrorMitigation

#--------------------Access circuits we want to run-----------
# Define accepted commands
VALID_EXPERIMENTS = ['2QEcho', '2QCumulative', 'Data']
CIRCS = Path("Circuits")
ART = Path("Circuits")
CIRCS_2QCumulative = Path("Circuits/2QCumulative/") 
CIRCS_2Qecho       = Path("Circuits/2QEcho/") 
CIRCS_Data_batch   = Path("Circuits/Data_batch/") 

OUT = Path("Execution_History_and_Status")
OUT.mkdir(exist_ok=True)

JOBS_SUBMITTED_LOG =  OUT/ "jobs_submitted.csv"
CONFIG = Path("Configurations")


# Define all config files
QUE                  = "experiment_queue.csv"
CONFIG_GENERAL       = CONFIG / "General_config.json"
CONFIG_API           = CONFIG / "API_config.json"

# ----------------------------------------------General configuration, used in all cases-----------------------
try:
    with open(CONFIG_GENERAL, "r") as f:
        cfg = json.load(f)
except FileNotFoundError:
        raise FileNotFoundError(f"Error: Configuration file '{CONFIG_GENERAL}' not found.")

try:
    with open(CONFIG_API, "r") as f:
        api_cfg = json.load(f)
except FileNotFoundError:
        raise FileNotFoundError(f"Error: Configuration file '{CONFIG_API}' not found.")

qubit_mapping = cfg.get("QUBIT_MAPPING")
token = api_cfg.get("IONQ_API_KEY")

if "IONQ_API_KEY" not in api_cfg:
        raise KeyError("Error: 'IONQ_API_KEY' is required in the configuration file.")

# Validate token is not empty
if not token or not str(token).strip():
    raise ValueError("Error: 'IONQ_API_KEY' cannot be empty.")

# Get backend_name with fallback
backend_name = cfg.get("BACKEND_NAME")
noise_model = cfg.get("NOISE_MODEL")

# Validate backend_name is not empty
if not backend_name or not str(backend_name).strip():
    raise ValueError("Error: 'BACKEND_NAME' cannot be empty.")

#-----------------------------------------------------specific config -----------------------------------------


def load_config_experiment_from_queue(row):
    """
    Extracts experiment_type, shots, qubit_config, and theta_alpha from a CSV queue row.

    Expected fields:
        Experiment_Type:   "2QEcho", "2QCumulative", or "Data"
        shots:             int
        qpair:             JSON list (e.g. [[5,6],[7,8]]) for 2Q experiments
        theta_alpha:       "0.5_1.2" for Data
    """

    # ----- EXPERIMENT TYPE -----
    experiment = (row.get("Experiment_Type") or "").strip()
    if experiment == "":
        raise ValueError("Error: 'Experiment_Type' is missing in queue row.")

    # ----- SHOTS -----
    shots_value = row.get("shots")
    if shots_value is None or str(shots_value).strip() == "":
        raise ValueError("Error: 'shots' cannot be empty.")

    try:
        shots = int(shots_value)
        if shots <= 0:
            raise ValueError("Error: 'shots' must be positive.")
    except Exception:
        raise ValueError(f"Invalid integer for 'shots': {shots_value}")
    debiasing_value = row.get("debiasing")
    val_str = str(debiasing_value).strip().lower() if debiasing_value is not None else ""

    if val_str == "true":
        debiasing_value = True
    elif val_str == "false" or val_str == "":
        debiasing_value = False
    else:
        raise ValueError(f"Invalid boolean value for 'debiasing': {debiasing_value}")

    # ==============================================
    #  CHAR EXPERIMENTS (2QEcho, 2QCumulative)
    # ==============================================
    if experiment in ["2QEcho", "2QCumulative"]:
        qpair_str = (row.get("qpair") or "").strip()
        if qpair_str == "":
            raise KeyError(f"Error: 'qpair' required for {experiment}.")

        # Parse JSON list of lists
        try:
            qubit_config = json.loads(qpair_str)
        except Exception as e:
            raise ValueError(f"Error parsing qpair JSON '{qpair_str}': {e}")

        # Validate structure
        if not isinstance(qubit_config, list):
            raise ValueError("qpair must be a list of lists.")

        for pair in qubit_config:
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ValueError(f"Invalid qubit pair: {pair}")
            if not all(isinstance(x, int) and x >= 0 for x in pair):
                raise ValueError(f"Qubit indices must be non-negative ints: {pair}")

        theta_alpha = "-"   # placeholder for 2Q exps

    # ==============================================
    #  DATA EXPERIMENT
    # ==============================================
    elif experiment == "Data":
        theta_alpha = (row.get("theta_alpha") or "").strip()
        if theta_alpha == "":
            raise KeyError("Error: 'theta_alpha' required for Data experiment.")

        qubit_config = "-"  # Data uses global qubit_mapping only

    # ==============================================
    #  UNKNOWN EXPERIMENT TYPE
    # ==============================================
    else:
        raise ValueError(f"Unknown Experiment_Type '{experiment}' in CSV row.")

    return experiment, shots, qubit_config, theta_alpha,debiasing_value

def ensure_jobs_header(path: Path):
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["submitted_utc","backend","noise_model", "experiment","circ_name","job_id","qubit_mapping","Qubit_Pair", "shots","Theta_alpha", "circuit_file"])
def write_results(experiment,backend_name,noise_model,job_id,shots,qubit_pair,circ_name,qpy_path,theta_alpha):
    with open(JOBS_SUBMITTED_LOG, "a", newline="") as f_out:
        
        w = csv.writer(f_out)
        
            
        
       
        w.writerow([
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            backend_name,
            noise_model,
            experiment,
            circ_name,            
            job_id,
            qubit_mapping,
            qubit_pair,
            shots,
            theta_alpha,
            qpy_path
        ])
        
        # optional tiny pause to be polite
        time.sleep(0.2)
def load_and_run_qpy_circuits(experiment,folder_path, backend_name,noise_model,token, shots,qubit_pairs,theta_alpha,debiasing = False):
    """
    Load all .qpy circuit files from a folder and run them.
    
    Args:
        folder_path: Path to folder containing .qpy files
        token: IonQ API token
        backend_name: Backend to run on
        shots: Number of shots
    """
    
    try:
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Error: Folder '{folder_path}' does not exist.")
        
        if not folder.is_dir():
            raise NotADirectoryError(f"Error: '{folder_path}' is not a directory.")
        
        # Get all .qpy files
        qpy_files = sorted(folder.glob("*.qpy"), key=lambda p: p.name)
        
        if not qpy_files:
            print(f"No .qpy files found in '{folder_path}'")
            return
        
        
        if(experiment == '2QEcho' or experiment == '2QCumulative'):
            if(len(qpy_files)>1):
                raise ValueError("Error: Characterization run only expects 1 circuit file in folder. Check for duplicates.")
            
            for pair in qubit_pairs:
                
                try:
                    print(f"\nProcessing: {qpy_files[0].name}")
                    #print(pair)
                    
                    # Load circuit from QPY file
                    with open(qpy_files[0], 'rb') as f:
                        from qiskit.qpy import load
                        circuits = load(f)

                    remapped_circuits = []
                    for i in range(len(circuits)):
                        #This is done because our .qpy file for characterization will include the circuit for each angle
                        circuit = circuits[i] if isinstance(circuits, list) else circuits                     

                        remapped_circ = QuantumCircuit(36, circuit.num_clbits)
                        remapped_circ.compose(circuit, qubits=pair, inplace=True)
                        remapped_circ.measure_all()
                        remapped_circuits.append(remapped_circ)
                        
                    
                    provider = IonQProvider(token=token)  # IonQ API token
                    backend_used = provider.get_backend(backend_name,gateset = 'native')
                    if(backend_name == "simulator"):
                        job = backend_used.run(remapped_circuits, shots=shots,noise_model = noise_model )
                    else:
                        job = backend_used.run(remapped_circuits, shots=shots,error_mitigation=ErrorMitigation.NO_DEBIASING)
                        noise_model = None
                    job_id = job.job_id()
                    write_results(experiment,backend_name,noise_model,job_id,shots,pair,qpy_files[0].name,qpy_files[0],theta_alpha)

                    print(f"Submitted: {qpy_files[0].name} : job_id={job_id}")

                except Exception as e:
                    print(f"Error processing {qpy_files[0].name}: {e}")
                    continue
                
        elif(experiment == 'Data'):
       
            for qpy_file in qpy_files:
                try:
                    print(f"\nProcessing: {qpy_file.name}")

                    # Load circuit from QPY file
                    with open(qpy_file, 'rb') as f:
                        from qiskit.qpy import load
                        circuits = load(f)
                        circuit = circuits[0] if isinstance(circuits, list) else circuits

                    
                    qubit_mapping
                    
                    remapped_circ = QuantumCircuit(36, circuit.num_clbits)
                    remapped_circ.compose(circuit, qubits=qubit_mapping, inplace=True)
                    remapped_circ.measure_all()
                    # Run the circuit (add your execution logic here)
                    provider = IonQProvider(token=token)  # IonQ API token

                    backend = provider.get_backend(backend_name,gateset = 'native')
                    # print(backend_name)
                    if(backend_name == "simulator"):
                        job = backend.run(remapped_circ, shots=shots,noise_model = noise_model )
                    else:
                        if(debiasing == False):
                            job = backend.run(remapped_circ, shots=shots,error_mitigation=ErrorMitigation.NO_DEBIASING)
                        if(debiasing == True):
                            circuit.measure_all()
                            job = backend.run(circuit, shots=shots,error_mitigation=ErrorMitigation.DEBIASING)
                        noise_model = None
                    job_id = job.job_id()
                    pair = "-"
                    write_results(experiment,backend_name,noise_model,job_id,shots,pair,qpy_file.name,qpy_file,theta_alpha)

                    print(f"Submitted: {qpy_file.name} : job_id={job_id}")

                except Exception as e:
                    print(f"Error processing {qpy_file.name}: {e}")
                    continue
                
                
        
    except Exception as e:
        raise Exception(f"Error loading QPY circuits: {e}")
    

def load_next_from_queue(path: Path):
    """
    Uses boolean mask (submitted == False) to find the next pending row.
    Returns (row_index, row_dict) or (None, None) if everything is done.
    """

    df = pd.read_csv(path)

    # Ensure pandas interprets this column as boolean even if CSV had strings
    if df["submitted"].dtype != bool:
        df["submitted"] = df["submitted"].astype(bool)

    pending_df = df[df["submitted"] == False]

    if pending_df.empty:
        return None, None

    # Get first pending row
    first_idx = pending_df.index[0]
    row_dict = pending_df.loc[first_idx].to_dict()

    return first_idx, row_dict

def mark_queue_row_submitted(path: Path, row_index: int):
    """
    Mark the queue row at row_index as submitted='true'.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"Row index {row_index} is out of range")

    rows[row_index]["submitted"] = True

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
def main():

    idx,row = load_next_from_queue(QUE)

    experiment, shots, qubit_pairs,theta_alpha,debiasing_value = load_config_experiment_from_queue(row)
    
    debiasing_value = False
    

    print(f"Selected experiment: {experiment}")

    if experiment == "Data":
        print(f" → theta_alpha = {theta_alpha}")
        
    else:
        print(" → no extra parameters required")
    # Command is valid, proceed with experiment



    print(f"Executing experiment: {experiment}")
    ensure_jobs_header(JOBS_SUBMITTED_LOG)

     # Load configuration and run corresponding circuit
    if experiment == '2QEcho':
        
       
        
        print(token)
        print(backend_name)
        print(type(shots))  
        print(qubit_pairs[0])  
         
        print(theta_alpha)
        load_and_run_qpy_circuits(experiment,CIRCS_2Qecho,backend_name,noise_model,token,shots,qubit_pairs,theta_alpha)
        mark_queue_row_submitted(QUE, idx)
    elif experiment == '2QCumulative':
        
        print(token)
        print(backend_name)
        print(shots) 
        print(qubit_pairs) 
        print(theta_alpha) 
        
        load_and_run_qpy_circuits(experiment,CIRCS_2QCumulative,backend_name,noise_model,token,shots,qubit_pairs,theta_alpha)
        mark_queue_row_submitted(QUE, idx)
        pass
        
    elif experiment == 'Data':
        folder_path = CIRCS_Data_batch / f"Data_{theta_alpha}"
        load_and_run_qpy_circuits(experiment,folder_path,backend_name,noise_model,token,shots,qubit_pairs,theta_alpha,debiasing_value)
        mark_queue_row_submitted(QUE, idx)
        # Do something for command 'c'
        pass

if __name__ == "__main__":
    main()


