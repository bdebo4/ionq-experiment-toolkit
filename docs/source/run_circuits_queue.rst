run\_circuits\_queue module
===========================


Purpose
-------

``run_circuits_queue.py`` submits experiments from the experiment queue CSV file.

It reads the queue, loads the requested QPY circuit files, submits the circuits
to the selected IonQ backend, and writes submitted job information to the jobs
log.

Main workflow
-------------

1. Read the experiment_queue.csv.
2. Parse each experiment row.
3. Load the corresponding QPY circuit files.
4. Remap circuits to the requested qubits.
5. Submit jobs to the configured backend.
6. Save submitted job IDs and metadata to jobs_submitted.csv.
7. Update the row in experiment_queue.csv as submitted.

Important functions
-------------------

``load_config()``
    Loads the API key and backend name from the configuration files.

``load_config_experiment_from_queue(row)``
    Parses one row from the experiment_queue.csv and returns the experiment settings.

``load_and_run_qpy_circuits(...)``
    Loads QPY circuits, remaps them, submits them to IonQ, and logs the jobs.

``write_results(...)``
    Appends one submitted job record to the jobs log CSV.

