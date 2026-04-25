# workflow.py
import sys, csv, json, time, os, pickle, ast, qiskit
import shutil
from typing import Dict, List
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Analysis_Tools.Extract_Info_From_IonQ_Json import load_json_innerkeys_binary, sum_by_bit_condition_dict
from Analysis_Tools.fidelity_calculation import data_to_pop_par, analyze_two_qubit_fidelity, raw_data_to_fidelity
import pandas as pd
from qiskit_ionq import GPIGate, GPI2Gate, ZZGate
from qiskit import QuantumCircuit
from qiskit.qpy import load as qpy_load
from qiskit_ionq import IonQProvider

save_fidelity_to_csv = True

provider = IonQProvider()
backend_native = provider.get_backend("simulator", gateset="native")

API_CONFIG = Path("Configurations/API_config.json")
with open(API_CONFIG, "r") as f:
    api_cfg = json.load(f)
os.environ.pop("IONQ_API_KEY", None)
os.environ["IONQ_API_KEY"] = api_cfg["IONQ_API_KEY"]
os.getenv('IONQ_API_KEY')

provider = IonQProvider()
backend = provider.get_backend("qpu.forte-1", gateset="native")

STATUS_OUT     = Path("Execution_History_and_Status/job_status.csv")
JOBS_SUBMITTED = Path("Execution_History_and_Status/jobs_submitted.csv")
FIDELITY_OUT = Path("Execution_History_and_Status/fidelities_time.csv")
# FIDELITY_OUT.mkdir(exist_ok=True)
# ----------------------------

# raw_data_loaded = load_json_innerkeys_binary(
#     RAW_FILEPATH, bit_width=36)


# test_job_id = "019aa190-789c-745d-b64f-a9271bec74cd"
# Data analysis functions ----------------------------

def measured_data(job_id: str) -> Dict[str, float]:
    """
    Given an IonQ job object, retrieve and return the counts dictionary.
    """

    job = backend.retrieve_job(job_id)
    result = job.get_probabilities()
    return result


def reattach_ionq_gates(circ: QuantumCircuit) -> QuantumCircuit:
    """
    After qpy_load (or JSON load), IonQ-native gates may come back as generic
    Instructions named 'gpi', 'gpi2', 'zz' without matrices. Replace them
    with the real qiskit_ionq GPIGate/GPI2Gate/ZZGate so Statevector can
    simulate them, while correctly preserving qubits and clbits.
    """

    # Create a new circuit with the same qubits and clbits (same sizes)
    new = QuantumCircuit(circ.num_qubits, circ.num_clbits)

    # (Optional) preserve metadata / name
    new.metadata = getattr(circ, "metadata", None)
    new.name = circ.name

    # Build maps from old bits -> new bits so we don't mix circuits
    qubit_map = {old_q: new.qubits[i] for i, old_q in enumerate(circ.qubits)}
    clbit_map = {old_c: new.clbits[i] for i, old_c in enumerate(circ.clbits)}

    for instr in circ.data:
        # Qiskit >= 1.2: instr is a CircuitInstruction
        try:
            op = instr.operation
            qargs = instr.qubits
            cargs = instr.clbits
        except AttributeError:
            # Backward compatibility if needed (inst, qargs, cargs)
            op, qargs, cargs = instr

        # Reattach IonQ native gates
        if op.name == "gpi":
            phase = float(op.params[0])     # phase in turns
            op = GPIGate(phase)
        elif op.name == "gpi2":
            phase = float(op.params[0])
            op = GPI2Gate(phase)
        elif op.name == "zz":
            angle = float(op.params[0])
            op = ZZGate(angle)
        # else: leave op as-is (rx, ry, measure, etc.)

        # Map old bits to bits in the new circuit
        mapped_qargs = [qubit_map[q] for q in qargs]
        mapped_cargs = [clbit_map[c] for c in cargs]

        new.append(op, mapped_qargs, mapped_cargs)

    return new

def load_qisk_and_map_from_job_id(job_id: str):
    """
    Given a job_id, look it up in jobs_submitted.csv and load:
        - the Qiskit circuit associated with that job_id
        - the Mapping (list of integers)

    Returns:
        (QuantumCircuit, mapping_list)

    The returned QuantumCircuit has IonQ native gates reattached as
    qiskit_ionq.GPIGate / GPI2Gate / ZZGate, so you can safely call
    qiskit.quantum_info.Statevector(...) on it.
    """

    # 1. Read the jobs_submitted.csv file
    df = pd.read_csv(JOBS_SUBMITTED)

    # 2. Find row for this job_id
    matches = df[df["job_id"] == job_id]
    if matches.empty:
        raise ValueError(f"Job ID '{job_id}' not found in {JOBS_SUBMITTED.name}")

    row = matches.iloc[0]

    # --- Extract circuit file path ---
    if "circuit_file" not in row:
        raise ValueError("Column 'circuit_file' missing from {JOBS_SUBMITTED.name}")

    # circuit_path = Path(row["circuit_file"])
    circuit_path_str = str(row["circuit_file"])
    circuit_path_str = circuit_path_str.replace("\\", os.sep)
    circuit_path = Path(circuit_path_str)

    # If it's a relative path, anchor it to this script's directory
    if not circuit_path.is_absolute():
        circuit_path = (Path(__file__).resolve().parent / circuit_path).resolve()


    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

    # --- Extract Mapping ---
    if "qubit_mapping" not in row:
        raise ValueError("Column 'qubit_mapping' missing from {JOBS_SUBMITTED.name}")

    mapping_raw = row["qubit_mapping"]

    # Parse mapping safely
    if isinstance(mapping_raw, str):
        try:
            mapping = ast.literal_eval(mapping_raw)  # e.g. "[0,1,2,3,...]"
        except Exception:
            raise ValueError(f"Could not parse Mapping from string: {mapping_raw}")
    elif isinstance(mapping_raw, (list, tuple)):
        mapping = list(mapping_raw)
    else:
        raise ValueError(f"Mapping is in unknown format: {mapping_raw}")

    # Ensure mapping is a list of integers
    mapping = list(mapping)
    if not all(isinstance(x, int) for x in mapping):
        raise ValueError(f"Mapping contains non-integer values: {mapping}")

    # --- Load the circuit (always Qiskit format) ---
    suffix = circuit_path.suffix.lower()

    if suffix == ".qpy":
        with open(circuit_path, "rb") as f:
            circuits = qpy_load(f)
            circuit = circuits[0] if isinstance(circuits, list) else circuits

    elif suffix == ".json":
        with open(circuit_path, "r") as f:
            circ_dict = json.load(f)
        circuit = QuantumCircuit.from_dict(circ_dict)

    else:
        raise RuntimeError(f"Unsupported circuit file format: {suffix}")

    # 🔑 Reattach IonQ gate classes so Statevector can handle them
    circuit = reattach_ionq_gates(circuit)

    return circuit, mapping

# Helper functions ---------------------------------

def calculate_ideal_populations(qiskit_circuit, bit_order: str = "right_to_left"):
    """
    Compute the ideal bitstring population distribution for a Qiskit circuit.

    Returns a dict: {bitstring: probability}.

    bit_order:
        "right_to_left" -> use Qiskit's default convention: leftmost bit is q_{n-1}
        "left_to_right" -> reverse the bitstrings so leftmost bit is q_0
                           (for compatibility with other conventions).
    """
    qc = qiskit_circuit.copy()
    sv = qiskit.quantum_info.Statevector(qc)
    probs_dict = sv.probabilities_dict()  # keys like "000...0"

    if bit_order == "right_to_left":
        # Directly use Qiskit's bitstring convention
        return dict(probs_dict)
    elif bit_order == "left_to_right":
        # Flip each bitstring so leftmost char corresponds to qubit 0
        return {k[::-1]: v for k, v in probs_dict.items()}
    else:
        raise ValueError("bit_order must be 'right_to_left' or 'left_to_right'")


# print(calculate_ideal_populations(qisk))

def fidelity_from_populations(ideal_populations, measured_populations):
    """
    Calculate fidelity given ideal and measured populations. Returns a float.
    """
    if len(ideal_populations) != len(measured_populations):
        raise ValueError("Ideal and measured populations must have the same length.")

    fidelity = np.sum(np.sqrt(ideal_populations * measured_populations)) ** 2
    return fidelity

def remap_ion_probs_to_qubit_probs(
    ion_result: Dict[str, float],
    mapping: List[int],
    renormalize: bool = True,
    bit_order: str = "right_to_left",
) -> Dict[str, float]:
    """
    Remap hardware ion probabilities to logical qubit probabilities.

    Parameters
    ----------
    ion_result : dict
        Dictionary mapping 36-bit ion strings -> probability.
        Example key: '000000...000' (length = n_ions).
    mapping : list of int
        Mapping from logical qubit index to ion index.
        Example: mapping = [21, 10, 2, 33, 22, 27, 35, 8, 31, 15]
        means:
            qubit 0 <- ion 21
            qubit 1 <- ion 10
            ...
    renormalize : bool, default True
        If True, renormalize the output probabilities so they sum to 1
        (recommended if you are discarding info from ions not in mapping).
    bit_order : {"left_to_right", "right_to_left"}, default "right_to_left"
        How to interpret the bitstring indices:
            - "left_to_right": bitstring[i] is ion i (leftmost bit is q_0)
            - "right_to_left": bitstring[-1 - i] is ion i (rightmost bit is q_0)

    Returns
    -------
    qubit_result : dict
        Dictionary mapping len(mapping)-bit qubit strings -> probability.
    """
    if not isinstance(mapping, (list, tuple, np.ndarray)):
        raise TypeError(
            f"mapping must be a list/tuple/array of ints, got {type(mapping)}: {mapping}"
        )
    if not ion_result:
        return {}

    # Infer the number of ions from any key
    example_key = next(iter(ion_result.keys()))
    n_ions = len(example_key)

    # Basic sanity checks
    if any(len(k) != n_ions for k in ion_result.keys()):
        raise ValueError("All bitstring keys in ion_result must have the same length.")

    if max(mapping) >= n_ions or min(mapping) < 0:
        raise ValueError(
            f"Mapping indices must be between 0 and {n_ions-1}; got {mapping}."
        )

    if bit_order not in ("left_to_right", "right_to_left"):
        raise ValueError("bit_order must be 'left_to_right' or 'right_to_left'.")

    qubit_result: Dict[str, float] = {}

    for ion_bits, p in ion_result.items():
        # Build the qubit bitstring according to mapping
        qubit_bits = []
        for q_idx, ion_idx in enumerate(mapping):
            if bit_order == "left_to_right":
                bit = ion_bits[ion_idx]
            else:  # "right_to_left"
                bit = ion_bits[-1 - ion_idx]
            qubit_bits.append(bit)

        qubit_key = "".join(qubit_bits)
        
        # If using right_to_left order, reverse the qubit bitstring to match Qiskit's convention
        # where rightmost bit is q_0 (least significant qubit)
        if bit_order == "right_to_left":
            qubit_key = qubit_key[::-1]

        # Accumulate probabilities for qubit patterns
        qubit_result[qubit_key] = qubit_result.get(qubit_key, 0.0) + p

    # Optional renormalization
    if renormalize:
        total = sum(qubit_result.values())
        if total > 0:
            for k in qubit_result:
                qubit_result[k] /= total

    return qubit_result

def compute_pauli_expectations(measure_dict):
    """
    Computes the expectation value for each p_string in measure_dict.
    For each tuple key (e.g., (+1,-1,+1)) inside measure_dict[p_string], multiply
    the product of its elements by its associated probability (value), and sum over all tuples.

    Returns:
        A dictionary: {p_string: expectation_value}
    """
    exp_dict = {}
    for p_string, outcomes in measure_dict.items():
        expectation_value = 0
        for output_tuple, prob in outcomes.items():
            prod = 1
            for num in output_tuple:
                prod *= num
            expectation_value += prod * prob
        exp_dict[p_string] = expectation_value
    return exp_dict


from itertools import product

def expectation_of_pauli_string_from_probs(measure_data, p_string):
    """
    Compute ⟨P⟩ for a Pauli string p_string using measurement data stored as
    eigenvalue-count dictionaries rather than per-shot lists.

    Parameters
    ----------
    measure_data : dict
        Keys: Pauli strings as tuples, e.g. ('X','Y','Z','X').
        Values: dict mapping eigenvalue tuples -> counts, e.g.
            {
              ( 1,  1,  1,  1): 47.0,
              ( 1,  1,  1, -1): 71.0,
              ...
            }

        This is exactly the structure you showed for measure_boundary.

    p_string : iterable of str
        Pauli labels, e.g. ('X','I','Z','Y'). May contain 'I'.

    Returns
    -------
    exp_val : float
        The expectation value ⟨P⟩.

    Notes
    -----
    - If p_string has no 'I', we compute:
          ⟨P⟩ = ( Σ_outcomes [ (∏ eigenvalues) * count ] ) / ( Σ_outcomes count )
    - If p_string contains 'I's, we mimic your original expectation_of_pauli_string:
        * Find indices of 'I' positions.
        * For each way of replacing "I" with X/Y/Z:
              p_ext ∈ {X,Y,Z}^#I
          we:
            - Look up measure_data[p_ext] (dict eigenvalues -> counts)
            - For each eigenvalue tuple, delete the 'I' positions
              before taking the product.
            - Compute ⟨P_ext_reduced⟩ as weighted average with counts.
        * Return the average of these expectations over all extensions.
    """
    p_string = tuple(p_string)

    # Case 1: no 'I' → directly compute expectation for this Pauli setting
    if 'I' not in p_string:
        if p_string not in measure_data:
            raise KeyError(f"No measurement data for Pauli string {p_string}")
        eig_counts = measure_data[p_string]  # dict: eigenvalue_tuple -> count

        total_counts = sum(eig_counts.values())
        if total_counts == 0:
            return 0.0

        num = 0.0
        for eigvals, count in eig_counts.items():
            # eigvals is e.g. (±1, ±1, ±1, ±1)
            prod_eig = 1.0
            for ev in eigvals:
                prod_eig *= ev
            num += prod_eig * count

        return num / total_counts

    # Case 2: p_string has 'I' → mimic original extension / averaging logic
    indices = [i for i, x in enumerate(p_string) if x == 'I']
    k = len(indices)
    exp_val_list = []

    for extend in product(['X', 'Y', 'Z'], repeat=k):
        # Build extended Pauli string, replacing 'I' positions with X/Y/Z
        p_string_extended = list(p_string)
        for idx, pauli_label in zip(indices, extend):
            p_string_extended[idx] = pauli_label
        p_string_extended = tuple(p_string_extended)

        if p_string_extended not in measure_data:
            raise KeyError(
                f"No measurement data for extended Pauli string {p_string_extended}"
            )

        eig_counts_ext = measure_data[p_string_extended]

        # Now compute expectation for the reduced operator where we
        # DROP the 'I' positions from the eigenvalue product.
        total_counts_ext = sum(eig_counts_ext.values())
        if total_counts_ext == 0:
            exp_val_list.append(0.0)
            continue

        num_ext = 0.0
        for eigvals, count in eig_counts_ext.items():
            # eigvals is a full tuple (length n)
            # remove the 'I' positions before taking the product
            reduced_eigs = [
                ev for idx_ev, ev in enumerate(eigvals) if idx_ev not in indices
            ]
            prod_eig = 1.0
            for ev in reduced_eigs:
                prod_eig *= ev
            num_ext += prod_eig * count

        exp_val_ext = num_ext / total_counts_ext
        exp_val_list.append(exp_val_ext)

    if len(exp_val_list) == 0:
        return 0.0

    # Average over all X/Y/Z extensions of the I's
    return sum(exp_val_list) / len(exp_val_list)

def circuit_fidelity(measured_populations, ideal_populations):
    import numpy as np
    keys = set(measured_populations.keys()) | set(ideal_populations.keys())
    bc = sum(np.sqrt(measured_populations.get(k, 0.0) *
                     ideal_populations.get(k, 0.0)) for k in keys)
    return bc**2


def get_jobs_to_plot(df_status):
    """
    Return DataFrame of jobs that need plotting 
    i.e. (Saved == True) & (Plotted == False).
    """
    jobs_to_plot = df_status[(df_status['Saved'] == True) & (df_status['Plotted'] == False)].copy()
    return jobs_to_plot

def get_parameters(df):
    """
    Extract parameters from a job dataframe.
    Expected columns:
        - Experiment   (str)
        - shots        (int)
        - Qubit_Pair   (str or list-like, only for 2Q experiments)
        - Mapping      (list-like of length 10, only for Experiment == "Data"

    Returns:
        dict with keys: Experiment, shots, q1, q2, Mapping
    """
    import re
    import ast
    import numpy as np

    # Use the first row of the dataframe (assuming one job per df)
    row = df.iloc[0]

    # Experiment and shots
    experiment = str(row["Experiment"])
    shots = int(row["shots"])

    # ---------- Qubit_Pair handling ----------
    q1 = None
    q2 = None

    if experiment in ["2QEcho", "2QCumulative"] and "Qubit_Pair" in df.columns:
        qp = row["Qubit_Pair"]

        # Case 1: list / tuple / numpy array
        if isinstance(qp, (list, tuple, np.ndarray)):
            if len(qp) >= 2:
                q1 = int(qp[0])
                q2 = int(qp[1])

        # Case 2: string (e.g. "1-3", "1,3", "[1, 3]", "q1=1 q2=3")
        elif isinstance(qp, str):
            # First, try to interpret it as a literal list/tuple, e.g. "[1, 3]"
            try:
                qp_eval = ast.literal_eval(qp)
                if isinstance(qp_eval, (list, tuple, np.ndarray)) and len(qp_eval) >= 2:
                    q1 = int(qp_eval[0])
                    q2 = int(qp_eval[1])
                else:
                    # Fallback: regex
                    nums = re.findall(r"\d+", qp)
                    if len(nums) >= 2:
                        q1 = int(nums[0])
                        q2 = int(nums[1])
            except (ValueError, SyntaxError):
                # Fallback: regex
                nums = re.findall(r"\d+", qp)
                if len(nums) >= 2:
                    q1 = int(nums[0])
                    q2 = int(nums[1])

    # ---------- Mapping handling ----------
    mapping = None
    if experiment == "Data" and "Mapping" in df.columns:
        mapping = row["Mapping"]

        # If it comes in as a string representation, try to parse it
        if isinstance(mapping, str):
            try:
                mapping_parsed = ast.literal_eval(mapping)
                # Only accept if it's list-like
                if isinstance(mapping_parsed, (list, tuple)):
                    mapping = list(mapping_parsed)
            except (ValueError, SyntaxError):
                # If parsing fails, just leave mapping as the original string
                pass

    return {
        "Experiment": experiment,
        "shots": shots,
        "q1": q1,
        "q2": q2,
        "Mapping": mapping,
    }

# Main routine -------------------------------------

def main(polling_frequency: int = 10):
    """
    Main routine to find jobs whose results have been saved but not yet plotted,
    load their raw data, and generate plots.

    The polling_frequency argument is kept for API symmetry with other workflows,
    but is not currently used in this one-shot plotting routine.
    """



    # # Path to THIS script
    # SCRIPT_DIR = Path(__file__).resolve().parent

    # # Path to Raw_Data folder
    # DATA_TYPE    = "2QCumulative"
    # RAW_FILENAME = "test019a9ca5-d2f1-7288-a283-253b87ebce32.json"

    # RAW_DATA_DIR = SCRIPT_DIR / "Results" / DATA_TYPE / "Raw_Data"
    # SAVE_DATA_DIR = SCRIPT_DIR / "Results" / DATA_TYPE / "_figures_from_data"

    # RAW_FILEPATH = RAW_DATA_DIR / RAW_FILENAME



    # Load status CSV
    df_status = pd.read_csv(STATUS_OUT)

    # Get jobs that need plotting: (Saved == True) & (Plotted == False)
    jobs_to_plot = get_jobs_to_plot(df_status)

    if jobs_to_plot.empty:
        print("No jobs need plotting. Exiting.")
        return

    for idx, job_row in jobs_to_plot.iterrows():
        job_id = job_row["job_id"]
        print(f"Processing job_id: {job_id}")

        # Extract parameters for this job
        params = get_parameters(job_row.to_frame().T)

        # Construct file path to raw data
        # raw_filepath = Path(job_row["File_Path"])
        raw_path_str = str(job_row["Data_file_path"])
        raw_path_str = raw_path_str.replace("\\", os.sep)
        raw_filepath = Path(raw_path_str)

        if not raw_filepath.exists():
            print(f"[ERROR] Raw data file not found: {raw_filepath}")
            continue

        # Load raw data (bit_width chosen to match your experiment)
        raw_data_loaded = load_json_innerkeys_binary(raw_filepath, bit_width=36)

        # Process based on experiment type
        if params["Experiment"] in ("2QEcho", "2QCumulative"):

            # Call raw_data_to_fidelity with q_i, q_j and numshots from CSV
            raw_data_to_fidelity(
                raw_data_loaded,
                i=params["q1"],
                j=params["q2"],
                numdata=3,
                numshots=params["shots"],
                num_echo = np.array([10,20]),
                label=f"Qubits ({params['q1']},{params['q2']})",
                grouping="echo-major",
                plot=True,
                save_plot=True,
                save_dir=raw_filepath.parent.parent / "_figures_from_data",
                save_basename=f"Q{params['q1']}_{params['q2']}_fidelity",
                save_format="svg",
            )

        elif params["Experiment"] == "Data":
            # --- 1. Load ideal circuit + mapping from jobs_submitted.csv ---
            try:
                qisk_circ, mapping = load_qisk_and_map_from_job_id(job_id)
            except Exception as e:
                print(f"[ERROR] Could not load circuit/mapping for job {job_id}: {e}")
                continue

            # --- 2. Get measured hardware probabilities from IonQ backend ---
            try:
                ion_probs = measured_data(job_id)
            except Exception as e:
                print(f"[ERROR] Could not retrieve measured data for job {job_id}: {e}")
                continue

            # --- 3. Remap 36-ion probabilities -> 10 logical qubits ---
            try:
                # Use bit_order='right_to_left' to match Qiskit's bitstring convention
                measured_populations = remap_ion_probs_to_qubit_probs(
                    ion_result=ion_probs,
                    mapping=mapping,
                    renormalize=True,
                    bit_order="right_to_left",
                )
            except Exception as e:
                print(f"[ERROR] Could not remap ion probabilities for job {job_id}: {e}")
                continue

            # --- 4. Compute ideal populations from the Qiskit circuit ---
            try:
                ideal_populations = calculate_ideal_populations(
                    qisk_circ,
                    bit_order="right_to_left",  # match keys with measured_populations
                )
            except Exception as e:
                print(f"[ERROR] Could not compute ideal populations for job {job_id}: {e}")
                continue

            # --- 5. Classical (Bhattacharyya) fidelity between distributions ---
            try:
                fid = circuit_fidelity(measured_populations, ideal_populations)
            except Exception as e:
                print(f"[ERROR] Could not compute fidelity for job {job_id}: {e}")
                continue

            print(f"Job {job_id}, == Fidelity={fid:.6f} ==")
            print('')


        # Mark this job as plotted and persist to CSV
        df_status.at[idx, "Plotted"] = True
        df_status.to_csv(STATUS_OUT, index=False)
        print(f"Finished processing job_id: {job_id}")

if __name__ == "__main__":
    main()