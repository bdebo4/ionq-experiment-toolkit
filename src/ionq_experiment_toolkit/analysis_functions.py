import pandas as pd

# import packages
import cirq
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional
from typing import List, Dict
import matplotlib.pyplot as plt
from numpy.linalg import norm
# import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import GPIGate, GPI2Gate, ZZGate
from qiskit.quantum_info import state_fidelity
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit_aer import AerSimulator
from qiskit.converters import circuit_to_dag, dag_to_circuit
import qutip as qt
from collections import Counter
from itertools import product
from collections import defaultdict
from pathlib import Path, PureWindowsPath
import json



def cirq_to_qiskit(cirq_circuit):
    """
    Convert a Cirq circuit into an equivalent Qiskit quantum circuit.

    Parameters
    ----------
    cirq_circuit
        Cirq circuit to convert. The circuit may contain common single-qubit gates,
        common two-qubit gates, rotation gates, and global phase gates.

    Returns
    -------
    QuantumCircuit
        Qiskit circuit with the same number of qubits and the supported operations
        translated into Qiskit gate calls.

    Notes
    -----
    Supported gates include ``H``, ``X``, ``Y``, ``Z``, ``S``, ``CNOT``, ``CZ``,
    ``rx``, ``ry``, ``rz``, ``XXPowGate``, ``YYPowGate``, ``ZZPowGate``, and
    ``GlobalPhaseGate``. Unsupported gates are skipped and reported with a warning.
    """
    cirq_qubits = sorted(cirq_circuit.all_qubits())
    n_qubits = len(cirq_qubits)
    qiskit_circuit = QuantumCircuit(n_qubits)
    qubit_map = {cirq_q: i for i, cirq_q in enumerate(cirq_qubits)}

    for moment in cirq_circuit.moments:
        for op in moment.operations:
            gate = op.gate
            qubits = [qubit_map[q] for q in op.qubits]

            # Common named gates
            if gate == cirq.H:
                qiskit_circuit.h(qubits[0])
            elif gate == cirq.X:
                qiskit_circuit.x(qubits[0])
            elif gate == cirq.Y:
                qiskit_circuit.y(qubits[0])
            elif gate == cirq.Z:
                qiskit_circuit.z(qubits[0])
            elif gate == cirq.S:
                qiskit_circuit.s(qubits[0])
            elif gate == cirq.CNOT:
                qiskit_circuit.cx(qubits[0], qubits[1])
            elif gate == cirq.CZ:
                qiskit_circuit.cz(qubits[0], qubits[1])

            # Rotation gates
            elif isinstance(gate, cirq.rx(np.pi).__class__):  # XPowGate
                angle = gate.exponent * np.pi
                qiskit_circuit.rx(angle, qubits[0])
            elif isinstance(gate, cirq.ry(np.pi).__class__):  # YPowGate
                angle = gate.exponent * np.pi
                qiskit_circuit.ry(angle, qubits[0])
            elif isinstance(gate, cirq.rz(np.pi).__class__):  # ZPowGate
                angle = gate.exponent * np.pi
                qiskit_circuit.rz(angle, qubits[0])

            # Two-qubit rotation gates
            elif isinstance(gate, cirq.XXPowGate):
                angle = gate.exponent * np.pi
                qiskit_circuit.rxx(angle, qubits[0], qubits[1])
            elif isinstance(gate, cirq.YYPowGate):
                angle = gate.exponent * np.pi
                qiskit_circuit.ryy(angle, qubits[0], qubits[1])
            elif isinstance(gate, cirq.ZZPowGate):
                angle = gate.exponent * np.pi
                qiskit_circuit.rzz(angle, qubits[0], qubits[1])

            # Global phase
            elif isinstance(gate, cirq.GlobalPhaseGate):
                qiskit_circuit.global_phase += np.angle(gate.coefficient)
            else:
                print(f"Warning: Unsupported gate type {type(gate).__name__}: {gate}")
    return qiskit_circuit



# Functions to simulate circuits ----------------------------------------------------

def get_qiskit_statevector(qiskit_circuit):
    """Get state vector using AerSimulator"""
    # Add save_statevector instruction
    circuit_copy = qiskit_circuit.copy()
    circuit_copy.save_statevector()
    
    # Run simulation
    simulator = AerSimulator(method='statevector')
    job = simulator.run(circuit_copy, shots=1)
    result = job.result()
    return result.get_statevector()

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

def plot_population_for_states(measured_vector, title):
    """
    Plot the population distribution over computational basis states.

    Parameters
    ----------
    measured_vector : array-like
        Vector containing the measured or simulated population for each basis
        state. The length of the vector determines the number of states shown.
    title : str
        Title displayed above the plot.

    Returns
    -------
    None
        Displays a bar plot of the population distribution.

    Notes
    -----
    This function is intended for an 8-qubit system, where the first and last
    x-axis labels correspond to ``|00000000⟩`` and ``|11111111⟩``.
    """
# Plot the simulated measurement output from cirq circuits for all 2^8 possible states.
    width = 0.6
    br1 = np.arange(len(measured_vector)) 
    # br2 = [x + width for x in br1] 
    fig = plt.subplots(figsize =(16, 3)) 
    plt.bar(br1, measured_vector, color ='r', width = width, label ='data') 
    # plt.bar(br2, measured_vector, color ='b', width = width, label ='expected population')
    plt.title(title)
    plt.xlabel('state')
    plt.ylabel('Population')
    plt.legend()
    plt.xticks([0,255],['|00000000⟩','|11111111⟩'], rotation=0)


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

def reduced_entropy_a_from_statevector(statevector_full, a):
    n = int(np.log2(len(statevector_full)))
    keep = [a]
    trace_out = [i for i in range(n) if i not in keep]
    
    density_matrix_full = statevector_to_density_matrix(statevector_full)
    density_mat_reduced = partial_trace(density_matrix_full, trace_out)
    S = von_neumann_entropy(density_mat_reduced)
    return S



def get_qiskit_statevector(qiskit_circuit):
    """Get state vector using AerSimulator"""
    # Add save_statevector instruction
    circuit_copy = qiskit_circuit.copy()
    circuit_copy.save_statevector()
    
    # Run simulation
    simulator = AerSimulator(method='statevector')
    job = simulator.run(circuit_copy, shots=1)
    result = job.result()
    return result.get_statevector()

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

def plot_population_for_states(measured_vector, title):
# Plot the simulated measurement output from cirq circuits for all 2^8 possible states.
    width = 1.0
    br1 = np.arange(len(measured_vector)) 
    # br2 = [x + width for x in br1] 
    fig = plt.subplots(figsize =(16, 3)) 
    plt.bar(br1, measured_vector, color ='r', width = width, label ='data') 
    # plt.bar(br2, measured_vector, color ='b', width = width, label ='expected population')
    plt.title(title)
    plt.xlabel('state')
    plt.ylabel('Population')
    plt.legend()
    plt.xticks([0,2**10-1],['|00000000⟩','|11111111⟩'], rotation=0)
    # plt.bar(np.linspace(0,31,32), expected_pop)
    # plt.bar(np.linspace(0,31,32), data, width=0.3)

def reduced_probs_from_statevector_numpy(psi: np.ndarray, keep: list[int], nq: int) -> np.ndarray:
    """
    function written by chatgpt on 2/16/2026
    psi: statevector length 2^nq, assumed in the same qubit order you want to interpret (you already call reverse_qubit_order)
    keep: qubit indices to keep, in the bit-order you want for returned probs (LSB is keep[0] convention assumed by your pipeline).
    Returns probs array length 2^k.
    """
    keep = list(keep)
    k = len(keep)

    probs_full = np.abs(psi)**2
    # reshape to n qubits tensor (axis 0 corresponds to most-significant in lex ordering)
    tensor = probs_full.reshape([2]*nq)

    # We want to sum over traced-out qubits
    trace_out = [q for q in range(nq) if q not in keep]
    # Move keep axes to the end (or beginning) in a consistent order
    # We'll move keep axes to the last k axes in the order given by keep.
    perm = trace_out + keep
    tensor_perm = np.transpose(tensor, axes=perm)

    # Sum over traced-out axes
    if len(trace_out) > 0:
        tensor_red = tensor_perm.reshape(2**len(trace_out), 2**k).sum(axis=0)
    else:
        tensor_red = tensor_perm.reshape(2**k)

    # tensor_red is ordered with keep axes as trailing axes in the order given by keep, using row-major flattening
    return tensor_red.astype(float)


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

def expectation_of_pauli_string_from_probs(measure_data, p_string):
    """
    Compute the expectation value of a Pauli string from eigenvalue counts.
    
    Parameters
    ----------
    measure_data : dict
        Dictionary mapping Pauli-string tuples to outcome-count dictionaries.
    p_string : iterable of str
        Pauli string containing ``"I"``, ``"X"``, ``"Y"``, and ``"Z"``.
    
    Returns
    -------
    float
        Expectation value of the Pauli string.
    
    Notes
    -----
    If ``p_string`` contains identity operators, those positions are averaged over
    all corresponding ``X``, ``Y``, and ``Z`` measurement bases.
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

def bootstrap_w_replacement(probs_dict, number_of_repeats=100, rng=None):    
    """
    Perform bootstrap resampling with replacement on a dictionary of counts.

    Parameters
    ----------
    probs_dict : dict
        Dictionary mapping bitstrings to counts.
    number_of_repeats : int, optional
        Number of bootstrap resamples to generate. Default is 100.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    dict
        Bootstrapped counts with the same bitstring keys. The total count is
        ``sum(probs_dict.values()) * number_of_repeats``.
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


# ---------- Circuit Conversion functions ----------
TAU = 2 * np.pi
PI  = np.pi
TOL = 1e-9

def _mod_turns(x):
    y = x % 1.0
    return 0.0 if abs(y - 1.0) < 1e-12 else y

def _turns(radians):
    return float(radians) / TAU

def _near(x, target, tol=TOL):
    return abs(x - target) <= tol

def _normalize_pi(theta):
    t = (theta + PI) % TAU - PI
    for v in (0.0, 0.5*PI, -0.5*PI, PI, -PI):
        if abs(t - v) <= 1e-12:
            return float(v)
    return float(t)

def to_ionq_gpi(circ: QuantumCircuit,
                phase_turns_global=None,
                global_qubits=None):
    """
    Convert a Qiskit circuit in {rx, ry, rz, rzz} basis to IonQ's {gpi, gpi2, zz} basis,
    using a moving Z frame tracked in a *global* phase vector.

    Parameters
    ----------
    circ : QuantumCircuit
        Input circuit in RX/RY/RZ/RZZ basis (num_qubits = n_local).
    phase_turns_global : list[float] or None
        Global phase frame in *turns*, indexed by physical qubit (e.g. 0..9).
        Will be extended with zeros if too short. If None, initialized to zeros
        up to max(global_qubits)+1.
    global_qubits : list[int] or None
        Mapping from local qubit index in `circ` to global physical qubit index.
        Length must equal circ.num_qubits.
        If None, defaults to identity [0, 1, ..., n_local-1].

    Returns
    -------
    new : QuantumCircuit
        Equivalent circuit in GPI/GPI2/ZZ basis, same num_qubits as `circ`.
    phase_turns_global : list[float]
        Updated global phase vector (same object shape, but possibly extended).
    """
    nq = circ.num_qubits
    new = QuantumCircuit(circ.num_qubits, circ.num_clbits)

    # Local->global qubit mapping
    if global_qubits is None:
        global_qubits = list(range(nq))
    elif len(global_qubits) != nq:
        raise ValueError(
            f"global_qubits length {len(global_qubits)} "
            f"does not match circuit.num_qubits={nq}"
        )

    # Ensure global phase vector is long enough and mutable
    max_idx = max(global_qubits) if global_qubits else -1
    if phase_turns_global is None:
        phase_turns_global = [0.0] * (max_idx + 1)  # one entry per physical qubit
    else:
        phase_turns_global = list(phase_turns_global)
        if len(phase_turns_global) <= max_idx:
            phase_turns_global.extend(
                [0.0] * (max_idx + 1 - len(phase_turns_global))
            )

    def emit_gpi2(i_local, base_turns):
        gq = global_qubits[i_local]
        ph = _mod_turns(base_turns + phase_turns_global[gq])
        new.append(GPI2Gate(ph), [new.qubits[i_local]])

    def emit_gpi(i_local, base_turns):
        gq = global_qubits[i_local]
        ph = _mod_turns(base_turns + phase_turns_global[gq])
        new.append(GPIGate(ph), [new.qubits[i_local]])

    for inst, qargs, cargs in circ.data:
        name = inst.name.lower()
        qs_local = [circ.find_bit(qb).index for qb in qargs]
        qs_global = [global_qubits[i] for i in qs_local]

        # ----- RZ: absorb as virtual Z by rotating the phase frame -----
        if name == "rz":
            theta = float(inst.params[0])   # radians
            t = _turns(theta)               # θ / 2π in turns
            for gq in qs_global:
                phase_turns_global[gq] = _mod_turns(phase_turns_global[gq] - t)
            continue

        # ----- RX handling -----
        if name == "rx":
            i_local = qs_local[0]
            gq = global_qubits[i_local]
            theta = _normalize_pi(float(inst.params[0]))

            if _near(theta,  0.5*PI):     # +π/2
                emit_gpi2(i_local, 0.0)
            elif _near(theta, -0.5*PI):   # -π/2
                emit_gpi2(i_local, 0.5)
            elif _near(theta,  PI):       # +π
                emit_gpi(i_local, 0.0)
            elif _near(theta, -PI):       # -π
                emit_gpi(i_local, 0.5)
            elif _near(theta, 0.0):       # 0 -> no-op
                pass
            else:
                # General-angle Rx(θ):
                emit_gpi2(i_local, 0.75)  # first GPI2
                phase_turns_global[gq] = _mod_turns(
                    phase_turns_global[gq] - _turns(theta)
                )  # virtual Rz(θ)
                emit_gpi2(i_local, 0.25)  # second GPI2
            continue

        # ----- RY handling -----
        if name == "ry":
            i_local = qs_local[0]
            gq = global_qubits[i_local]
            theta = _normalize_pi(float(inst.params[0]))

            if _near(theta,  0.5*PI):     # +π/2
                emit_gpi2(i_local, 0.25)
            elif _near(theta, -0.5*PI):   # -π/2
                emit_gpi2(i_local, 0.75)
            elif _near(theta,  PI):       # +π
                emit_gpi(i_local, 0.25)
            elif _near(theta, -PI):       # -π
                emit_gpi(i_local, 0.75)
            elif _near(theta, 0.0):       # 0 -> no-op
                pass
            else:
                # General-angle Ry(θ):
                emit_gpi2(i_local, 0.0)  # first GPI2
                phase_turns_global[gq] = _mod_turns(
                    phase_turns_global[gq] - _turns(theta)
                )  # virtual Rz(θ)
                emit_gpi2(i_local, 0.5)  # second GPI2
            continue

        # ----- RZZ (radians) -> ZZ (turns) -----
        if name == "rzz":
            theta = float(inst.params[0])        # radians
            turns = _turns(theta)                # θ / (2π)
            q0_local, q1_local = qs_local
            gq0, gq1 = global_qubits[q0_local], global_qubits[q1_local]

            turns = _fold_zz_turns_into_range(turns, phase_turns_global, gq0, gq1)

            new.append(ZZGate(turns), [new.qubits[q0_local], new.qubits[q1_local]])
            continue
            #  Commented out to use the new function _fold_zz_turns_into_range on 2/12/2026
            # theta = float(inst.params[0])        # radians
            # turns = _turns(theta)                # θ / (2π)
            # q0_local, q1_local = qs_local
            # new.append(
            #     ZZGate(turns),
            #     [new.qubits[q0_local], new.qubits[q1_local]],
            # )
            # continue

        # ----- ZZ already in turns: pass through (remap bits) -----
        if name == "zz":
            q0_local, q1_local = qs_local
            gq0, gq1 = global_qubits[q0_local], global_qubits[q1_local]

            turns = float(inst.params[0])
            turns = _fold_zz_turns_into_range(turns, phase_turns_global, gq0, gq1)

            new.append(ZZGate(turns), [new.qubits[q0_local], new.qubits[q1_local]])
            continue
            #  Commented out to use the new function _fold_zz_turns_into_range on 2/12/2026
            # q0_local, q1_local = qs_local
            # new.append(inst, [new.qubits[q0_local], new.qubits[q1_local]], [])
            # continue

        # ----- Other gates: remap and append as-is -----
        new_qargs = [new.qubits[i] for i in qs_local]
        if cargs:
            cs = [circ.find_bit(cb).index for cb in cargs]
            new_cargs = [new.clbits[i] for i in cs]
        else:
            new_cargs = []
        new.append(inst, new_qargs, new_cargs)

    return new, phase_turns_global

def _fold_zz_turns_into_range(turns: float, phase_turns_global, gq0: int, gq1: int):
    """
    Fold ZZ 'turns' into [-0.25, 0.25] by using the identity:
      ZZ(t + 0.5)  ~  (Z ⊗ Z) · ZZ(t)     (up to global phase)

    Since Z is virtual in your framework, we implement (Z ⊗ Z) by updating the
    moving phase frame: Rz(pi) on each qubit => subtract 0.5 turns.
    """
    # Fold into [-0.25, 0.25)
    folded = ((turns + 0.25) % 0.5) - 0.25

    # How many half-turns did we remove?
    # (Use rounding to remind floating error; okay for numeric constants.)
    k = int(np.round((turns - folded) / 0.5))

    # If k is odd, we owe a (Z ⊗ Z) which we do virtually via phase frame update.
    if (k % 2) != 0:
        phase_turns_global[gq0] = _mod_turns(phase_turns_global[gq0] - 0.5)
        phase_turns_global[gq1] = _mod_turns(phase_turns_global[gq1] - 0.5)

    return folded

def cirq_to_qisk_ionq(cirq_circuit, opt_level, phase_turns_global=None):
    """
    Convert a Cirq circuit to an IonQ-native Qiskit circuit with GPI/GPI2/ZZ gates,
    while updating a *global* per-physical-qubit phase frame.

    Parameters
    ----------
    cirq_circuit : cirq.Circuit
        Input Cirq circuit (operating on some subset of LineQubit.range(10)).
    opt_level : int
        Qiskit transpiler optimization level.
    phase_turns_global : list[float] or None
        Global phase frame (in turns) indexed by physical qubit (0..9).
        Can be None to start from all zeros; will be extended if needed.

    Returns
    -------
    qiskit_ionq_native : QuantumCircuit
        IonQ-native circuit (GPI/GPI2/ZZ) with num_qubits == number of qubits used.
    phase_turns_global : list[float]
        Updated global phase frame after this circuit.
    """
    # Figure out which physical qubits this Cirq circuit uses
    cirq_qubits = sorted(cirq_circuit.all_qubits())  # should be LineQubits
    global_qubits = [q.x for q in cirq_qubits]       # physical indices, e.g. [0,5] or [0..9]

    # Convert Cirq -> Qiskit in RX/RY/RZ/RZZ basis
    ions_gates = ['rx', 'ry', 'rz', 'rzz']
    qiskit_circuit = cirq_to_qiskit(cirq_circuit)  # your existing converter uses same ordering
    qiskit_circuit = transpile(qiskit_circuit,
                               basis_gates=ions_gates,
                               optimization_level=opt_level)

    # Now convert to IonQ-native, using global phase tracking
    qiskit_ionq_native, phase_turns_global = to_ionq_gpi(
        qiskit_circuit,
        phase_turns_global=phase_turns_global,
        global_qubits=global_qubits,
    )
    return qiskit_ionq_native, phase_turns_global

# Quantum state tomography (QST) functions ------------------------------------------------------------

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
    Generate Pauli basis strings and matrices for a subsystem.

    Parameters
    ----------
    qoi : list of int
        Qubit indices included in the subsystem.

    Returns
    -------
    tuple
        ``(basis_strings, basis_matrices)``, where ``basis_strings`` contains Pauli
        label tuples and ``basis_matrices`` contains the corresponding matrices.
        The all-identity string is excluded.
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
        # print('p_string:', p_string, ', exp_val: ', exp_val)
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




def state_reconstruction_from_probs(measure_data, qoi, eps=1e-12, use_mlm=True):
    """
    Reconstruct a reduced density matrix from Pauli-basis probabilities.
    
    Parameters
    ----------
    measure_data : dict
        Dictionary mapping Pauli-string tuples to bitstring-probability
        dictionaries.
    qoi : list of int
        Qubits of interest. Only ``len(qoi)`` is used.
    eps : float
        Eigenvalue cutoff used in the entropy calculation.
    use_mlm : bool
        Whether to apply maximal-likelihood projection.
    
    Returns
    -------
    tuple
        ``(rho, entropy)``, where ``rho`` is the reconstructed density matrix and
        ``entropy`` is the von Neumann entropy in bits.
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





def convert_to_ordered_list(prob_dict, num_qubits=10):
    """Return a list of probabilities indexed 0..2^n-1 in Qiskit ordering."""
    ordered = np.zeros(2**num_qubits)
    for bitstring, p in prob_dict.items():
        idx = int(bitstring, 2)  # Qiskit-consistent
        ordered[idx] = p
    return ordered

# Functions to calculate entropy ----------------------------------------------------



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