# poll_jobs.py
import json, csv, sys,time
import os
from datetime import datetime, timezone
from pathlib import Path
from qiskit_ionq import IonQProvider
from qiskit.providers.jobstatus import JobStatus
from qiskit_ibm_runtime import RuntimeEncoder
import ast
import pandas as pd 
from qiskit_ionq.exceptions import IonQJobFailureError


EXPERIMENTS = ['2QEcho', '2QCumulative', 'Data']

CONFIG_GENERAL = Path("Configurations/General_config.json")
API_CONFIG = Path("Configurations/API_config.json")
JOBS_SUBMITTED = Path("Execution_History_and_Status/jobs_submitted.csv")
STATUS_OUT = Path("Execution_History_and_Status/job_status.csv")
STATUS_OUT_COMPLETION_TIME = Path("Execution_History_and_Status/job_status.csv")
RESULTS_DIR = Path("Results")
# Create Results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)
try:
    with open(CONFIG_GENERAL, "r") as f:
        cfg = json.load(f)
except FileNotFoundError:
        raise FileNotFoundError(f"Error: Configuration file '{CONFIG_GENERAL}' not found.")
qubit_mapping = cfg.get("QUBIT_MAPPING")
experiment_paths = {}
for exp in EXPERIMENTS:
    raw_data_path = RESULTS_DIR / exp / "Raw_Data"
    raw_data_path.mkdir(parents=True, exist_ok=True)
    experiment_paths[exp] = raw_data_path


def load_config():
    with open(CONFIG_GENERAL, "r") as f:
        cfg = json.load(f)
    with open(API_CONFIG, "r") as f:
        api_cfg = json.load(f)
    return api_cfg["IONQ_API_KEY"], cfg.get("BACKEND_NAME", "simulator")

def ensure_status_header(path: Path):
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["checked_utc","Experiment","Status", "job_id","Qubit_Pair","Theta_alpha","Mapping","shots","Plotted","Saved","Data_file_path","Circ_file_path"])
# Now define the _save_result function
def _save_result(job, jid, circuit, backend, timestamp, characteristic, theta_alpha=None, Data_characteristic=None):
    """Save job.result() to experiment-specific folder and update job_status.csv."""
    
    # Determine which experiment folder to use
    if circuit in experiment_paths:
        save_dir = experiment_paths[circuit]
        if circuit == "Data" and theta_alpha is not None:
            save_dir = save_dir.parent / "Raw_Data"/f"Data_{theta_alpha}"
            save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = RESULTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output file path
    filename = f"{timestamp}_{circuit}_{characteristic}_job_id_{jid}.json"
    out = save_dir / filename

    try:
        job = backend.retrieve_job(jid)
        result = job.get_counts()
        payload = result
        
    except Exception as e:
        payload = {"error": str(e)}
        print(f"Error retrieving results for {jid}: {e}")
        return None

    try:
        with open(out, "w") as f:
            json.dump(payload, f, cls=RuntimeEncoder, indent=2)
        
        print(f"Saved result for {jid} → {out}")
        
        # Get job info from the status file to update properly
        df_status = pd.read_csv(STATUS_OUT)
        job_row = df_status[df_status['job_id'] == jid].iloc[0]
        
        # Update the row with saved status
        update_or_add_job_status(
            jid=jid,
            circtype=job_row['Experiment'],
            status_name=job_row['Status'],
            q_pair=job_row['Qubit_Pair'],
            
            theta_alpha=job_row['Theta_alpha'],
            saved=True,
            Data_file_path=str(out),
            Circ_file_path=job_row['Circ_file_path']
        )
        
        return str(out)
        
    except Exception as e:
        print(f"Error saving file for {jid}: {e}")
        return None

def sync_job_status_with_submitted():
    """
    Ensure all jobs from jobs_submitted.csv exist in job_status.csv.
    Maintains the same order as jobs_submitted.csv.
    Adds missing jobs with initial status.
    """
    # Read submitted jobs
    df_submitted = pd.read_csv(JOBS_SUBMITTED)
    
    # Read or create status file
    if STATUS_OUT.exists():
        df_status = pd.read_csv(STATUS_OUT)
    else:
        df_status = pd.DataFrame(columns=["checked_utc", "Experiment", "Status", "job_id", 
                                         "Qubit_Pair", "Theta_alpha", "Mapping","shots",
                                         "Plotted", "Saved",  "Data_file_path","Circ_file_path"])
    
    # Find jobs in submitted but not in status
    existing_job_ids = set(df_status['job_id'].tolist())
    new_jobs = []
    
    for _, row in df_submitted.iterrows():
        jid = row['job_id']
        if jid not in existing_job_ids:
            # Add new job with initial values
            new_jobs.append({
                "checked_utc": "",
                "Experiment": row['experiment'],
                "Status": "NOT_POLLED",
                "job_id": jid,
                "Qubit_Pair": row['Qubit_Pair'],
                
                "Theta_alpha": row['Theta_alpha'],
                "Mapping":qubit_mapping,
                "shots": row['shots'],
                "Plotted": False,
                "Saved": False,
                "Data_file_path": "",
                "Circ_file_path":row['circuit_file']
            })
    
    # Add new jobs to status DataFrame
    if new_jobs:
        df_new = pd.DataFrame(new_jobs)
        df_status = pd.concat([df_status, df_new], ignore_index=True)
        print(f"Added {len(new_jobs)} new jobs to job_status.csv")
    
    # Reorder df_status to match df_submitted order
    # Create a mapping of job_id to order in submitted
    submitted_order = {jid: idx for idx, jid in enumerate(df_submitted['job_id'])}
    
    # Add order column, sort, then remove it
    df_status['_order'] = df_status['job_id'].map(submitted_order)
    df_status = df_status.sort_values('_order').drop('_order', axis=1)
    
    # Save back to CSV
    df_status.to_csv(STATUS_OUT, index=False)
    
    return df_status
def get_jobs_to_poll(df_status):
    """
    Return DataFrame of jobs that still need polling.
    
    Conditions:
      - Saved == False
      - Final_State NOT in (CANCELLED, ERROR)
    """
    final_states_to_exclude = {
        JobStatus.CANCELLED.name,
        JobStatus.ERROR.name
    }
    
    jobs_to_poll = df_status[
        (df_status["Saved"] == False) &
        (~df_status["Status"].isin(final_states_to_exclude))
    ].copy()
    
    return jobs_to_poll

def update_or_add_job_status(jid, circtype, status_name, q_pair, theta_alpha, shots=None,saved=False,Data_file_path="",Circ_file_path =""):
    """Update existing job entry or add new one using pandas. NO DUPLICATES."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Read existing status file or create empty DataFrame
    if STATUS_OUT.exists():
        df = pd.read_csv(STATUS_OUT)
        # Ensure correct dtypes for string columns
        df['checked_utc'] = df['checked_utc'].astype(str)
        df['Data_file_path'] = df['Data_file_path'].astype(str)
        
        # Check if job exists
        if jid in df['job_id'].values:
            # UPDATE existing row - don't add new one
            df.loc[df['job_id'] == jid, 'checked_utc'] = timestamp
            df.loc[df['job_id'] == jid, 'Status'] = status_name
            df.loc[df['job_id'] == jid, 'Saved'] = saved
            if Data_file_path:
                df.loc[df['job_id'] == jid, 'Data_file_path'] = Data_file_path
        else:
            # Only add new row if job doesn't exist
            new_row = pd.DataFrame([{
                "checked_utc": timestamp,
                "Experiment": circtype,
                "Status": status_name,
                "job_id": jid,
                "Qubit_Pair": q_pair,
                
                "Theta_alpha": theta_alpha,
                "Mapping":qubit_mapping,
                "shots": shots,
                "Plotted": False,
                "Saved": saved,
                "Data_file_path": Data_file_path,
                "Circ_file_path": Circ_file_path
            }])
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        # File doesn't exist - create with first job
        df = pd.DataFrame([{
            "checked_utc": timestamp,
            "Experiment": circtype,
            "Status": status_name,
            "job_id": jid,
            "Qubit_Pair": q_pair,
            "Mapping":qubit_mapping,
            "shots": shots,
            "Theta_alpha": theta_alpha,
            "Plotted": False,
            "Saved": saved,
            "Data_file_path": Data_file_path,
            "Circ_file_path": Circ_file_path

        }])
    
    # Save back to CSV
    df.to_csv(STATUS_OUT, index=False)

def get_job_completed_time(backend, job_id: str, to_local_tz: bool = True):
    """
    Return the job's 'Completed' time as a Python datetime.

    Uses IonQ's job API via qiskit_ionq's internal client:
      - For API v0.4: uses 'completed_at' (ISO timestamp string).

    Parameters
    ----------
    backend : IonQBackend
        Backend returned by IonQProvider.get_backend(...).
    job_id : str
        IonQ job UUID.
    to_local_tz : bool
        If True (default), convert from UTC to the local timezone of the machine.

    Returns
    -------
    datetime | None
        Completed time as an aware datetime, or None if not available.
    """
    try:
        job = backend.retrieve_job(job_id)
    except Exception as e:
        print(f"[completed_time] Failed to retrieve job {job_id}: {e}")
        return None

    # Use the same client the IonQJob uses internally
    try:
        raw = job._client.retrieve_job(job._job_id)  # uses IonQClient under the hood
    except Exception as e:
        print(f"[completed_time] Failed to retrieve raw job JSON for {job_id}: {e}")
        return None

    # v0.4 API: completed_at is an ISO-8601 string, e.g. "2025-11-20T20:04:50.123Z"
    completed_iso = raw.get("completed_at")
    completed_dt = None

    if completed_iso:
        # Normalize trailing 'Z' to +00:00 for fromisoformat
        iso_str = completed_iso.replace("Z", "")
        # iso_str = completed_iso.replace("Z", "+00:00")

        # print('iso_str', iso_str)
        try:
            completed_dt = datetime.fromisoformat(iso_str)
        except ValueError as e:
            print(f"[completed_time] Could not parse completed_at '{completed_iso}' "
                  f"for job {job_id}: {e}")

    if completed_dt is None:
        # Nothing usable found
        print(f"[completed_time] No completed_at/response field for job {job_id}.")
        return None

    # Ensure we treat it as UTC if it somehow came in naive
    if completed_dt.tzinfo is None:
        completed_dt = completed_dt.replace(tzinfo=timezone.utc)

    if to_local_tz:
        completed_dt = completed_dt.astimezone()  # convert to local machine timezone
        completed_dt = completed_dt.replace(tzinfo=None)

    return completed_dt

def populate_completion_times_from_completion(poll_only_done: bool = True):
    """
    Populate the 'completion_time' column in job_status_time.csv with the job's
    COMPLETED time (as an ISO datetime string).

    - If job_status_time.csv does not exist, it is initialized from job_status.csv.
    - Adds an 'completion_time' column if missing.
    - For each job_id, calls get_job_completed_time(...) and writes the ISO string.

    Parameters
    ----------
    poll_only_done : bool
        If True, only query jobs whose Status == 'DONE'.
        If False, will attempt to query all jobs (jobs not yet completed will
        typically return None and be left blank).
    """
    # Connect to IonQ backend using your existing config
    token, backend_name = load_config()
    provider = IonQProvider(token=token)
    backend = provider.get_backend(backend_name)

    # Load the time-augmented status file, or fall back to the base status file
    if STATUS_OUT_COMPLETION_TIME.exists():
        df = pd.read_csv(STATUS_OUT_COMPLETION_TIME)
        print(f"[completed_time] Loaded existing {STATUS_OUT_COMPLETION_TIME}")
    else:
        df = pd.read_csv(STATUS_OUT)
        print(f"[completed_time] {STATUS_OUT_COMPLETION_TIME} not found; "
              f"starting from {STATUS_OUT}")

    # Ensure the column exists
    if "completion_time" not in df.columns:
        # df["completion_time"] = pd.NA
        df["completion_time"] = df.get("completion_time", pd.NA).astype("string")

    updated_count = 0

    for idx, row in df.iterrows():
        # Optionally only consider jobs that have finished
        if poll_only_done and row.get("Status") != JobStatus.DONE.name:
            continue

        # Skip if already filled
        current_val = df.at[idx, "completion_time"]
        if pd.notna(current_val) and str(current_val) != "":
            continue

        job_id = row["job_id"]

        completed_dt = get_job_completed_time(backend, job_id, to_local_tz=True)
        if completed_dt is None:
            continue

        # Store as ISO string so it's easy to parse later
        df.at[idx, "completion_time"] = completed_dt.isoformat()
        updated_count += 1

    df.to_csv(STATUS_OUT_COMPLETION_TIME, index=False)
    print(f"[completed_time] Updated completion_time (Completed timestamps) "
          f"for {updated_count} rows in {STATUS_OUT_COMPLETION_TIME}")

def main(poll_every_sec: int = 10):
    """Poll jobs and save results, updating status without duplicates."""
    
    token, backend_name = load_config()
    provider = IonQProvider(token=token)
    backend = provider.get_backend(backend_name)
    
    # Sync job_status.csv with jobs_submitted.csv
    print("Syncing job_status.csv with jobs_submitted.csv...")
    df_status = sync_job_status_with_submitted()
    
    # Get jobs that need polling (Saved == False)
    jobs_df = get_jobs_to_poll(df_status)
    
    if jobs_df.empty:
        print("All jobs have been saved. Nothing to poll.")
        return
    
    print(f"Tracking {len(jobs_df)} unsaved jobs on backend '{backend_name}'")
    
    # Poll jobs
    remaining = set(jobs_df['job_id'].tolist())
    
    while remaining:
        finished = []
        
        for jid in list(remaining):
            # Get job info from DataFrame
            job_row = jobs_df[jobs_df['job_id'] == jid].iloc[0]
            
            try:
                print("Polled successfully")
                job = backend.retrieve_job(jid)
                
                status = job.status()
                
                circtype = job_row['Experiment']
                q_pair = job_row['Qubit_Pair']
                
                
                theta_alpha = job_row['Theta_alpha']
                shots = job_row['shots']
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                print(f"{circtype}  |  {jid}  |  {status.name}")
                
                # Update status (this updates the existing row, no duplicates)
                update_or_add_job_status(jid, circtype, status.name, q_pair,  theta_alpha, shots,False, "")
                
                
                # If finished, save result
                if status is JobStatus.DONE:
                    df_submitted = pd.read_csv(JOBS_SUBMITTED)
                    submitted_row = df_submitted[df_submitted['job_id'] == jid].iloc[0]
                    submission_timestamp = submitted_row['submitted_utc']

                    if circtype in ["2QEcho", "2QCumulative"]:

                        qubit_pair_list = ast.literal_eval(str(q_pair))
                        characteristic = f"Q_{qubit_pair_list[0]}_{qubit_pair_list[1]}"
                        _save_result(job, jid, circtype, backend, submission_timestamp, characteristic)
                        
                    elif circtype == "Data":
                        characteristic = f"theta_alpha_{theta_alpha}"
                        _save_result(job, jid, circtype, backend, timestamp, characteristic,theta_alpha)
                
                # Mark as finished if in final state
                if status in (JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR):

                    finished.append(jid)
                    
            except Exception as e:
                print(f"Error polling job {jid}: {e}")
                
                circtype = job_row['Experiment']
                q_pair = job_row['Qubit_Pair']
                
                
                theta_alpha = job_row['Theta_alpha']
                shots = job_row['shots']
                update_or_add_job_status(
                jid,
                circtype,
                JobStatus.ERROR.name,   # mark failure
                q_pair,
                theta_alpha,
                shots,
                True,                    # Saved = True so we stop polling
                str(e)                   # Error message
                )
                finished.append(jid)
                continue
        
        # Remove finished jobs from remaining
        for jid in finished:
            remaining.discard(jid)
        
        # Wait before next poll
        if remaining:
            time.sleep(poll_every_sec)
    
    print("\nPolling complete!")

# TODO uncomment this to run the main function
if __name__ == "__main__":
    main()

# token, backend_name = load_config()
# provider = IonQProvider(token=token)
# backend = provider.get_backend(backend_name)
# test_job_id = "019aa433-9458-73a8-808c-88e841ca6cd8"

# TODO uncomment this to run update the completion times
# populate_completion_times_from_completion(poll_only_done = True)