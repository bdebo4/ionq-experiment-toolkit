# Functions.py
from typing import List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.linalg import norm
from itertools import product
from collections import defaultdict
from pathlib import Path, PureWindowsPath
import json
# import cirq
import qiskit
from collections import Counter



SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SI = np.array([[1, 0], [0, 1]], dtype=complex)

def pauli_matrix_from_string(p_string):
    """
    p_string: e.g. ['X','Z','I'] for len(qoi)=3
    returns the corresponding 2^n x 2^n matrix
    """
    mat = 1
    for p in p_string:
        if p == 'X':
            mat = np.kron(mat, SX)
        elif p == 'Y':
            mat = np.kron(mat, SY)
        elif p == 'Z':
            mat = np.kron(mat, SZ)
        elif p == 'I':
            mat = np.kron(mat, SI)
    return mat

def generate_pauli_basis(qoi):
    """
    qoi: list of qubit indices whose reduced density matrix we reconstruct.
    We return:
      basis_strings: list of Pauli label tuples, e.g. ('X','Z','I',...)
      basis_matrices: corresponding numpy matrices (excluding all-I string)
    """
    n = len(qoi)
    labels = ['I', 'X', 'Y', 'Z']
    basis_strings = []
    basis_matrices = []

    for p_string in product(labels, repeat=n):
        if all(p == 'I' for p in p_string):
            continue  # identity will be added separately as I/2^n
        mat = pauli_matrix_from_string(p_string)
        basis_strings.append(p_string)
        basis_matrices.append(mat)

    return basis_strings, basis_matrices
  
def append_Paulis(qoi,p_string, NUM_QUBITS):                 # basis change: for X or Y, rotate to Z before measuring
    gate=[]
    qubits=cirq.LineQubit.range(NUM_QUBITS)
    for idx, qubit in enumerate(qoi):
        p = p_string[idx]
        if p == 'X':
            gate.append(cirq.H(qubits[qubit]))
        elif p == 'Y':
            gate.append(cirq.rx(np.pi / 2).on(qubits[qubit]))
    return cirq.Circuit(gate)
    

def measure_Zbasis(qc,qoi, NUM_QUBITS):               # measure in Pauli Z basis on the qubits in qoi. Return the bitstring as measurement outcome.
    mes_list=[]
    statevector = qiskit.quantum_info.Statevector(qc).data
    statevector = reverse_qubit_order(statevector)
    mes_state=statevector.reshape(*[2 for i in range(NUM_QUBITS)])
    projs=[np.array([1,0]),np.array([0,1])]
    for qubit in qoi:
        prob_list=[norm(np.tensordot(ele.conj(),mes_state,axes=[0,qubit]))**2 for ele in projs]
        i=np.random.choice([0,1],p=prob_list)
        mes_result=(-1)**i
        mes_list.append(mes_result)
        mes_state=np.tensordot(projs[i].conj(),mes_state,axes=[0,qubit])
        order=list(range(NUM_QUBITS))[1:]
        order.insert(qubit,0)
        mes_state=np.tensordot(projs[i],mes_state,axes=0).transpose(*order)/prob_list[i]**0.5
    return np.array(mes_list) 


def expectation_of_pauli_string(measure_data,p_string):               
    """
    Compute expectation value of a Pauli string on qoi using measurement data
    p_string: tuple/list of ['I','X','Y','Z'] of length len(qoi)
    """
    p_string=tuple(p_string)
    if not 'I' in p_string:
        data=measure_data[p_string]
        exp_val=sum([np.prod(ele) for ele in data])/len(data)
        return exp_val
    indices = [i for i, x in enumerate(p_string) if x == 'I']
    k=len(indices)
    exp_val_list=[]
    for extend in product(['X','Y','Z'],repeat=k):
        p_string_extended=list(p_string)
        for i in range(k):
            p_string_extended[indices[i]]=extend[i]
        data=measure_data[tuple(p_string_extended)].copy()
        short_data=[np.delete(ele,indices) for ele in data]
        exp_val=sum([np.prod(ele) for ele in short_data])/len(short_data)
        exp_val_list.append(exp_val)
    return sum(exp_val_list)/len(exp_val_list)
    
    
def mlm_rho(mu):                   # maximal-likelihood correction algorithm, removing negative eigenvalue
    w, v = np.linalg.eigh(mu)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    i = len(w) - 1
    a = 0
    lambda_ = np.zeros_like(w)

    while i >= 0:
        if w[i] + a / (i + 1) >= 0:
            break
        lambda_[i] = 0
        a += w[i]
        i -= 1

    for j in range(i + 1):
        lambda_[j] = w[j] + a / (i + 1)

    rho = sum(lambda_[i] * np.outer(v[:, i], v[:, i].conj()) for i in range(len(lambda_)))
    return rho


def state_reconstruction(measure_data, qoi, eps=1e-12, use_mlm=True):
    """
    Reconstruct reduced density matrix on qoi using Pauli tomography:
      mu = (1/2^n) [ I + sum_i <P_i> P_i ]
    If use_mlm=True, project mu to the closest physical density matrix via mlm_rho.
    Returns (rho_phys, entropy).
    """
    n   = len(qoi)
    dim = 2**n

    basis_strings, basis_matrices = generate_pauli_basis(qoi)

    # tomographic estimate mu
    mu = np.zeros((dim, dim), dtype=complex)

    for p_string, P in zip(basis_strings, basis_matrices):
        exp_val = expectation_of_pauli_string(measure_data,p_string)
        mu += (exp_val / (2**n)) * P

    mu += (1.0 / (2**n)) * np.eye(dim, dtype=complex)

    # enforce Hermiticity and unit trace before MLM
    mu = 0.5 * (mu + mu.conj().T)
    tr = np.trace(mu).real
    if tr != 0:
        mu /= tr

    if use_mlm:
        rho = mlm_rho(mu)
    else:
        rho = mu

    # eigen-decomposition and entropy
    evals, _ = np.linalg.eigh(rho)
    evals = np.clip(evals.real, eps, None)
    evals /= np.sum(evals)
    entropy = -np.sum(evals * np.log2(evals))

    return rho, entropy

def state_reconstruction_from_probs(measure_data, qoi, eps=1e-12, use_mlm=True):
    """
    Reconstruct reduced density matrix on qoi using Pauli tomography,
    using probability distributions instead of shot lists.

    This is the probability-analogue of `state_reconstruction` that
    previously used simulated ±1 measurement outcomes.

    Differences from state_reconstruction:
      - measure_data[p_string] is now a dict: bitstring -> probability,
        where bitstring is over the *already-traced-out* tomography qubits.
      - qoi is just used to set n = len(qoi); the bitstrings are already
        reduced to those qubits.

    Parameters
    ----------
    measure_data : dict
        Keys: Pauli strings as TUPLES, e.g. ('X','Y','Z','X'),
              exactly the same objects you used as keys before
              (e.g. from boundary_basis_set).

        Values: dict mapping
            bitstring (e.g. "0101") -> probability

        Bitstrings are measurement outcomes in the Pauli basis specified
        by the key p_string. Because your circuits already include the
        appropriate pre-rotations for X/Y/Z, we can interpret:
            bit '0' -> eigenvalue +1
            bit '1' -> eigenvalue -1
        for ALL p_strings in measure_data.

    qoi : list[int]
        Indices of qubits of interest (only used for n = len(qoi)).

    eps : float
        Cutoff for eigenvalues in entropy calculation.

    use_mlm : bool
        If True, apply maximal-likelihood projection mlm_rho(mu).

    Returns
    -------
    rho : np.ndarray
        Reconstructed density matrix on the tomography qubits.

    entropy : float
        von Neumann entropy (bits).
    """
    n   = len(qoi)
    dim = 2**n

    # Use exactly the same Pauli basis as your measurement set
    basis_strings = list(measure_data.keys())  # tuples like ('X','Y','Z','X')

    # Build matrices for each measured Pauli string
    basis_matrices = [pauli_matrix_from_string(p_string) for p_string in basis_strings]

    # Tomographic estimate μ
    mu = np.zeros((dim, dim), dtype=complex)

    for p_string, P in zip(basis_strings, basis_matrices):

        probs = measure_data[p_string]   # dict: bitstring -> probability

        # Compute ⟨P⟩ from probabilities.
        # Because pre-rotations diagonalize P into Z,
        # we can use the same mapping for all bases:
        #     bit '0' -> eigenvalue +1
        #     bit '1' -> eigenvalue -1
        exp_val = 0.0
        for bitstring, p in probs.items():
            product = 1.0
            for b in bitstring:
                product *= (+1.0 if b == '0' else -1.0)
            exp_val += p * product

        mu += (exp_val / (2**n)) * P

    # Add identity component
    mu += (1.0 / (2**n)) * np.eye(dim, dtype=complex)

    # Enforce Hermiticity and trace-normalize
    mu = 0.5 * (mu + mu.conj().T)
    tr = np.trace(mu).real
    if tr != 0.0:
        mu /= tr

    rho = mlm_rho(mu) if use_mlm else mu

    # Eigen-decomposition & entropy
    evals, _ = np.linalg.eigh(rho)
    evals = np.clip(evals.real, eps, None)
    evals /= np.sum(evals)
    entropy = -np.sum(evals * np.log2(evals))

    return rho, entropy

def classical_fidelity(p, q, eps=1e-15):
    """
    Compute classical fidelity between two probability distributions.
    
    Fidelity is:
        F(p,q) = ( sum_i sqrt(p_i * q_i) )^2
    
    Parameters
    ----------
    p : array-like
        First distribution (e.g., measured_list)
    q : array-like
        Second distribution (e.g., ideal_list)
    eps : float
        Numerical cutoff to avoid sqrt of negative numbers.
    
    Returns
    -------
    float : fidelity value in [0,1]
    """

    # Convert to numpy arrays
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Check lengths
    if p.shape != q.shape:
        raise ValueError(f"Distributions have different lengths: {len(p)} vs {len(q)}")

    # Clip negative rounding errors
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)

    # Normalize if necessary
    sp = p.sum()
    sq = q.sum()
    if abs(sp - 1) > 1e-12:
        p = p / sp
    if abs(sq - 1) > 1e-12:
        q = q / sq

    # Compute fidelity
    bc = np.sum(np.sqrt(p * q + eps))  # Bhattacharyya coeff
    F = bc**2

    # Numerical range enforcement
    return float(max(0, min(1, F)))


def bootstrap_w_replacement(probs_dict, number_of_repeats=100, rng=None):    
    """
    Perform bootstrap resampling with replacement on a dictionary of counts.

    Parameters
    ----------
    probs_dict : dict
        Dictionary mapping bitstring -> count (number of times measured).
    number_of_repeats : int, optional
        How many bootstrap resamples to generate. Default is 100.
    rng : np.random.Generator, optional
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    bootstrapped_counts : dict
        Dictionary of the same keys (bitstrings), where the values are the
        total counts from all bootstrap resamples combined.
        The sum of the values will be:
            sum(probs_dict.values()) * number_of_repeats
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Extract bitstrings (keys) and their counts (values)
    bitstrings = list(probs_dict.keys())
    counts = np.array(list(probs_dict.values()), dtype=float)

    # 2. Compute total number of shots in the original data
    total_shots = counts.sum()
    if total_shots < 500:
        raise ValueError("Too few shots for bootstrapping, minimum is 500.")

    # 3. Turn counts into empirical probabilities for each bitstring
    #    These are the probabilities we will sample from.
    p = counts / total_shots

    # 4. In standard bootstrap, each resample has 'total_shots' draws.
    #    If we create 'number_of_repeats' independent resamples and then
    #    combine them, that's equivalent to a *single* multinomial draw
    #    with (total_shots * number_of_repeats) total draws from the same p.
    total_draws = int(total_shots * number_of_repeats)

    # 5. Draw 'total_draws' samples *with replacement* from the bitstrings,
    #    according to the empirical probabilities p.
    sampled_bitstrings = rng.choice(
        bitstrings,
        size=total_draws,
        replace=True,
        p=p
    )

    # 6. Count how many times each bitstring appears in the combined bootstrap
    #    samples using a Counter.
    counter = Counter(sampled_bitstrings)

    # 7. Build the output dictionary:
    #    - Same set of keys as the input
    #    - Values are the new counts from the bootstrap
    #    - Use float to mirror your original example
    bootstrapped_counts = {
        bs: float(counter.get(bs, 0.0)) for bs in bitstrings
    }
    # (Optional sanity check)
    # assert abs(sum(bootstrapped_counts.values()) - total_draws) < 1e-9

    return bootstrapped_counts

def reverse_qubit_order(vec):
    "Reverse qubit indexing of a statevector or probability array."
    vec = np.asarray(vec)
    n = int(np.log2(len(vec)))
    if 2**n != len(vec):
        raise ValueError("Length of input must be a power of 2.")

    # Generate bit-reversed indices
    indices = np.arange(len(vec))
    bitstrings = ((indices[:, None] >> np.arange(n)) & 1)  # shape: (2^n, n)
    reversed_bitstrings = bitstrings[:, ::-1]
    reversed_indices = reversed_bitstrings.dot(1 << np.arange(n))
    return vec[reversed_indices]

def reduced_rho_from_statevector_numpy(psi: np.ndarray, keep_indices, n_qubits: int):

    keep_indices = list(keep_indices)
    keep_set = set(keep_indices)
    trace_indices = [i for i in range(n_qubits) if i not in keep_set]

    # Reshape |psi> into tensor with one axis per qubit: (2,2,...,2)
    psi_t = psi.reshape((2,) * n_qubits)

    # Permute axes so kept qubits come first
    perm = keep_indices + trace_indices
    psi_perm = np.transpose(psi_t, axes=perm)

    dk = 2 ** len(keep_indices)           # dimension of kept subsystem
    dt = 2 ** (n_qubits - len(keep_indices))  # dimension of traced-out subsystem

    # View as a matrix: rows = kept basis, cols = traced basis
    psi_mat = psi_perm.reshape(dk, dt)

    # rho_keep = psi_mat * psi_mat^\dagger
    rho = psi_mat @ psi_mat.conj().T
    return rho

def von_neumann_entropy_from_rho(rho: np.ndarray, base=2) -> float:
    rho = 0.5 * (rho + rho.conj().T)  # hermitize for numerical stability
    evals = np.linalg.eigvalsh(rho).real
    evals = np.clip(evals, 0.0, 1.0)
    evals = evals[evals > 0]
    return float(-np.sum(evals * (np.log(evals) / np.log(base))))

# Functions to extract data from job_id and data_file_database ----------------------------------------

def job_ids_from_circuit_name(circuit_name: str, spreadsheet_destination):
    """
    Extract job IDs from a circuit name.
    
    Args:
        circuit_name: str, name of the circuit
        spreadsheet_destination: str or pd.DataFrame, either a file path to CSV or a pandas DataFrame

    Returns:
        job_ids: list[str], list of job IDs
    """
    
    # Detect if spreadsheet_destination is a string (file path) or DataFrame
    if isinstance(spreadsheet_destination, str):
        df = pd.read_csv(spreadsheet_destination)   # It's a file path - read the CSV file
    elif isinstance(spreadsheet_destination, pd.DataFrame):
        df = spreadsheet_destination     # It's already a DataFrame - use it directly   
    else:
        raise TypeError(
            f"spreadsheet_destination must be either a string (file path) or a pandas DataFrame, "
            f"got {type(spreadsheet_destination).__name__}"
        )
    
    # Filter rows where circ_name matches circuit_name
    matching_rows = df[df['circ_name'] == circuit_name]
    
    # Extract job_id column and convert to list
    job_ids = matching_rows['job_id'].tolist()
    
    return job_ids


# job_ids = job_ids_from_circuit_name(
#     circuit_name = "00_Data_withMagic_theta_0785_mu_027_bound_XXXX_rx010_ry020_rz000_xx027.qpy",
#     spreadsheet_destination = JOBS_LIST_DATABASE
# )

def data_details_from_job_id(job_id: list[str], data_file_database: pd.DataFrame, get_completion_time = False):
    """
    Extracts data run details from a job ID, like qubit_mapping and data file path.

    Args:
        job_id: List of job IDs to extract details for.
        data_file_database: A pandas DataFrame containing job status details (from job_status.csv).

    Returns:
        mappings: List of Mapping values (as lists of integers) corresponding to each job_id (in same order)
        file_paths: List of File_Path values corresponding to each job_id (in same order)
    """
    import ast
    
    mappings = []
    file_paths = []
    completion_times = []
    
    # Iterate through job_id list to maintain order
    for jid in job_id:
        # Find the row with matching job_id
        matching_row = data_file_database[data_file_database['job_id'] == jid]
        
        if not matching_row.empty:
            # Extract Mapping and File_Path from the first matching row
            mapping_str = matching_row.iloc[0]['Mapping']
            file_path = matching_row.iloc[0]['Data_file_path']
            if get_completion_time:
                completion_time = matching_row.iloc[0]['completion_time']
                completion_times.append(completion_time)

            
            # Convert Mapping string to list of integers
            if isinstance(mapping_str, str):
                try:
                    mapping = ast.literal_eval(mapping_str)
                    # Ensure it's a list of integers
                    if isinstance(mapping, list):
                        mapping = [int(x) for x in mapping]
                    else:
                        mapping = None
                except (ValueError, SyntaxError):
                    mapping = None
            elif isinstance(mapping_str, list):
                # Already a list, just ensure integers
                mapping = [int(x) for x in mapping_str]
            else:
                mapping = None
            
            mappings.append(mapping)
            file_paths.append(file_path)
        else:
            # If job_id not found, append None (or you could raise an error)
            mappings.append(None)
            file_paths.append(None)
    
    if get_completion_time:
        return mappings, file_paths, completion_times
    else:
        return mappings, file_paths
    
    


# Set this to your repo root if needed (or leave as "." if your notebook runs from repo root)
BASE_DIR = Path(".").resolve()

def normalize_path(p: str) -> Path:
    """
    Convert Windows-style paths (backslashes) to the current OS path,
    and resolve relative paths against BASE_DIR.
    """
    # If it looks like a windows path, convert it
    if isinstance(p, str) and ("\\" in p):
        p = str(PureWindowsPath(p))  # converts to a normalized windows path string
        # Now reinterpret that path on current OS as a POSIX-ish path
        # PureWindowsPath -> parts -> Path
        p = Path(*PureWindowsPath(p).parts)
    else:
        p = Path(p)

    # Resolve relative paths relative to BASE_DIR
    if not p.is_absolute():
        p = (BASE_DIR / p)

    return p

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

def circuit_details_to_qubit_probs(mappings: List[int], file_paths: List[str]):
    """ 
    Extracts qubit probabilities from a list of file paths and mapping.
    Needs to import remap_ion_probs_to_qubit_probs from data_analysis.py
    """
    qubit_probs_list = [] # list of qubit probs for each mapping
    qubit_probs_dict = {} # combines all qubit probs for all mappings (more relevant)
    for mapping, file_path in zip(mappings, file_paths):
        file_path = normalize_path(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # read probabilities from data
        qubit_probs = remap_ion_probs_to_qubit_probs(
            ion_result = data,
            mapping = mapping,
            renormalize = False,
            bit_order = "right_to_left",
        )
        qubit_probs_list.append(qubit_probs)
    
    # combine all qubit probs for all mappings
    for qubit_probs in qubit_probs_list:
        for key, value in qubit_probs.items():
            if key not in qubit_probs_dict:
                qubit_probs_dict[key] = value
            else:
                qubit_probs_dict[key] += value
    return qubit_probs_list, qubit_probs_dict


def reverse_bitstring_keys(input_dict):
    """
    Returns a new dictionary with the same values as input_dict,
    but with each key (bitstring) reversed in order.
    E.g., key '01234' becomes '43210'.
    """
    return {key[::-1]: value for key, value in input_dict.items()}


def add_pauli_measures_to_dict(p_string, measure_dict, qubit_probs):
    """
    Adds a new entry to measure_dict with the structure:
    measure_dict[p_string][output_list] = probability from qubit_probs

    The keys are lists of +1/-1, derived from the bitstring (0→+1, 1→-1).

    Args:
        p_string (str): Pauli string label (e.g. 'XYZ')
        measure_dict (dict): The dictionary to update.
        qubit_probs (dict): Dictionary mapping bitstrings to probabilities.
    """
    if p_string in measure_dict:
        raise ValueError(f"p_string '{p_string}' already exists in measure_dict.")
    measure_dict[p_string] = {}
    for key, value in qubit_probs.items():
        # Convert bitstring to list of +1/-1 ints
        mapped_list = [1 if bit == '0' else -1 for bit in key]
        # Use lists as keys is not possible. Instead, use tuple.
        measure_dict[p_string][tuple(mapped_list)] = value

# add_pauli_measures_to_dict(test_p_string, test_dict, example_single_qubit_probs)

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

def state_reconstruction_from_pauli_expectations(measure_data, qoi, eps=1e-12, use_mlm=True):
    """
    Modified function from Gong's code to use qubit_probs_dictionary instead of measure_data.
    Should work with expectation_of_pauli_string_from_probs().
    Same as state_reconstruction, but uses measure_data as
    dict[p_string][eigenvalue_tuple] = counts.
    """
    n   = len(qoi)
    dim = 2**n

    basis_strings, basis_matrices = generate_pauli_basis(qoi)

    mu = np.zeros((dim, dim), dtype=complex)

    for p_string, P in zip(basis_strings, basis_matrices):
        exp_val = expectation_of_pauli_string_from_probs(measure_data, p_string)
        mu += (exp_val / (2**n)) * P

    mu += (1.0 / (2**n)) * np.eye(dim, dtype=complex)

    mu = 0.5 * (mu + mu.conj().T)
    tr = np.trace(mu).real
    if tr != 0:
        mu /= tr

    rho = mlm_rho(mu) if use_mlm else mu

    evals, _ = np.linalg.eigh(rho)
    evals = np.clip(evals.real, eps, None)
    evals /= np.sum(evals)
    entropy = -np.sum(evals * np.log2(evals))

    return rho, entropy

def normalize_measured_distribution(measured_dict):
    total = sum(measured_dict.values())
    return {k: v/total for k, v in measured_dict.items()}

def convert_to_ordered_list(prob_dict, num_qubits=10):
    """Return a list of probabilities indexed 0..2^n-1 in Qiskit ordering."""
    ordered = np.zeros(2**num_qubits)
    for bitstring, p in prob_dict.items():
        idx = int(bitstring, 2)  # Qiskit-consistent
        ordered[idx] = p
    return ordered

# Functions to calculate entropy ----------------------------------------------------

def statevector_to_density_matrix(statevector: np.ndarray) -> np.ndarray:
    "Convert a statevector of length 2^n to a density matrix of shape (2^n, 2^n)"
    statevector = statevector.reshape(-1, 1)  # column vector
    density_matrix = statevector @ statevector.conj().T
    return density_matrix

def partial_trace(density_matrix: np.ndarray, traced_out: list) -> np.ndarray:
    "Compute reduced density matrix by tracing out specific subsystems"
    
    num_qubits = int(np.log2(density_matrix.shape[0]))
    if len(traced_out) >= num_qubits:
        raise ValueError("Number of indices to trace out must be less than total qubits.")

    rho_qobj = qt.Qobj(density_matrix, dims=[[2]*num_qubits, [2]*num_qubits])

    # Subsystems to keep = complement of traced_out
    keep = [i for i in range(num_qubits) if i not in traced_out]

    rho_reduced = rho_qobj.ptrace(keep)

    return rho_reduced.full()  # convert Qobj back to NumPy

def von_neumann_entropy(reduced_density_matrix: np.ndarray, base=2) -> float:
    "Compute von Neumann entropy given the density matrix"
    rho_qobj = qt.Qobj(reduced_density_matrix)
    return qt.entropy_vn(rho_qobj, base=base)

# TODO: this function should be decommissioned
def reduced_entropy_abc_from_statevector(statevector_full, a, b, c):
    """
    psi: length 2^n complex array (Qiskit ordering)
    a,b,c: qubit indices you want to keep (0 is LSB in Qiskit)
    returns: (rho_abc, S_vN base-2)
    """
    
    n = int(np.log2(len(statevector_full)))
    keep = sorted([a, b, c])
    trace_out = [i for i in range(n) if i not in keep]
    
    density_matrix_full = statevector_to_density_matrix(statevector_full)
    density_mat_reduced = partial_trace(density_matrix_full, trace_out)
    S = von_neumann_entropy(density_mat_reduced)
    return S

def reduced_entropy_from_statevector(statevector_full, a: np.ndarray):
    
    n = int(np.log2(len(statevector_full)))
    keep = sorted(a)
    trace_out = [i for i in range(n) if i not in keep]
    
    density_matrix_full = statevector_to_density_matrix(statevector_full)
    density_mat_reduced = partial_trace(density_matrix_full, trace_out)
    S = von_neumann_entropy(density_mat_reduced)
    return S


def reduced_probs_from_statevector(sv, qoi_indices):
    """
    sv           : qiskit.quantum_info.Statevector
    qoi_indices  : list/array of physical qubit indices (e.g. boundary_ind or bulk_indices)

    Returns:
        dict mapping bitstring over qoi_indices -> probability

    Assumes Qiskit bitstring convention: rightmost bit is qubit 0, etc.
    """
    probs_full = sv.probabilities_dict()  # full 10-qubit bitstrings -> probs
    num_qubits = int(round(np.log2(len(sv.data))))

    reduced = defaultdict(float)

    for full_bits, p in probs_full.items():
        # full_bits is a string of length num_qubits, rightmost bit = qubit 0
        # We want bits ordered according to qoi_indices
        bits_qoi = []
        for q in qoi_indices:
            # index from right since qubit 0 is rightmost bit
            bits_qoi.append(full_bits[num_qubits - 1 - q])
        reduced_bits = "".join(bits_qoi)
        reduced[reduced_bits] += p

    return dict(reduced)