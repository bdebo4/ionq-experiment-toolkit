watch\_ionq\_status module
==========================

Purpose
-------

``watch_ionq_status.py`` tracks submitted IonQ jobs, updates their status, and saves
completed job results.

Main workflow
-------------

1. Read submitted jobs from ``jobs_submitted.csv``.
2. Synchronize them with ``job_status.csv``.
3. Poll unfinished jobs from the configured IonQ backend.
4. Update each job status.
5. Save completed job results to the ``Results`` folder.
6. Optionally populate completion timestamps.

Important functions
-------------------

``load_config()``
    Loads the IonQ API key and backend name.

``ensure_status_header(path)``
    Creates the job_status.csv header if the file does not exist.

``sync_job_status_with_submitted()``
    Adds missing submitted jobs to the job_status.csv file.

``get_jobs_to_poll(df_status)``
    Selects jobs that still need polling.

``update_or_add_job_status(...)``
    Updates an existing job_status.csv row or adds a new one.

``get_job_completed_time(backend, job_id, to_local_tz=True)``
    Retrieves the completed timestamp for a job.

``populate_completion_times_from_completion(poll_only_done=True)``
    Fills completion times for completed jobs.

``main(poll_every_sec=10)``
    Polls jobs until all relevant jobs reach a final state.

