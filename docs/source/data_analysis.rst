data\_analysis module
=====================



Purpose
-------

``data_analysis.py`` processes completed experiment jobs and generates analysis
outputs such as fidelities and plots.

Main workflow
-------------

1. Read job status information from ``job_status.csv``.
2. Find jobs whose results have been saved but not plotted.
3. Load raw result data.
4. Process characterization jobs or data jobs depending on experiment type.
5. Compute fidelities or generate plots.
6. Mark processed jobs as plotted in ``job_status.csv``.

Important functions
-------------------

``measured_data(job_id)``
    Retrieves measured probabilities for an IonQ job.

``reattach_ionq_gates(circ)``
    Replaces generic IonQ-native gate instructions with qiskit-ionq gate objects.

``load_qisk_and_map_from_job_id(job_id)``
    Loads the circuit and qubit mapping associated with a submitted job.

``calculate_ideal_populations(qiskit_circuit, bit_order="right_to_left")``
    Computes ideal bitstring probabilities from a Qiskit circuit.

``fidelity_from_populations(ideal_populations, measured_populations)``
    Computes fidelity between ideal and measured population distributions.

``remap_ion_probs_to_qubit_probs(...)``
    Maps 36-ion probabilities to logical-qubit probabilities.

``circuit_fidelity(measured_populations, ideal_populations)``
    Computes classical fidelity between two probability distributions.

``get_jobs_to_plot(df_status)``
    Selects saved jobs that have not yet been plotted.

``get_parameters(df)``
    Extracts experiment parameters from a job-status row.

``main(polling_frequency=10)``
    Runs the plotting and analysis workflow.

