import pandas as pd
from qiskit_ionq import IonQProvider
from pathlib import Path
import sys
import json

# ----------------------------
# CONFIGURATION
# ----------------------------
JOBS_STATUS_FILE = Path("Execution_History_and_Status/job_status.csv")
API_CONFIG = Path("Configurations/API_config.json")
CONFIG_GENERAL = Path("Configurations/General_config.json")




def load_config():
    """
    Load the API key and backend name from the configuration files.

    Returns
    -------
    tuple[str, str]
        Tuple containing the API key and backend name. If ``BACKEND_NAME`` is not
        defined in the general configuration file, ``"simulator"`` is used as the
        default backend name.
    """
    with open(CONFIG_GENERAL, "r") as f:
        cfg = json.load(f)
    with open(API_CONFIG, "r") as f:
        api_cfg = json.load(f)
    return api_cfg["IONQ_API_KEY"], cfg.get("BACKEND_NAME", "simulator")
def cancel_jobs_after_index(start_index):
    """
    Cancel all IonQ jobs listed after a given row index in the job status file.

    The function reads the job_status.csv file, selects all rows after
    ``start_index``, extracts the corresponding job IDs, and attempts to cancel
    each job using the configured IonQ backend.

    Parameters
    ----------
    start_index : int
        Row index after which jobs should be cancelled. The job at
        ``start_index`` itself is not cancelled.

    Returns
    -------
    None
        Prints cancellation progress and error messages to the console.

    Notes
    -----
    The job status CSV file must contain a ``job_id`` column. Backend credentials
    and backend name are loaded using ``load_config``.
    """
    # Load status file
    df = pd.read_csv(JOBS_STATUS_FILE)

    if start_index < 0 or start_index >= len(df):
        print(f"Index {start_index} is out of range (0 to {len(df)-1}).")
        return

    # Subset rows from start_index to end
    df_subset = df.iloc[start_index+1:]

    # Extract job IDs
    job_ids = df_subset["job_id"].dropna().astype(str).tolist()

    if not job_ids:
        print("No job IDs found in the selected range.")
        return

    print(f"Found {len(job_ids)} job(s) to cancel starting from index {start_index}:")
    for jid in job_ids:
        print("  -", jid)

    token, backend_name = load_config()
    provider = IonQProvider(token=token)
    backend = provider.get_backend(backend_name,gateset = "native")
    

    # Cancel jobs one by one
    print("\nCancelling jobs...")
    for jid in job_ids:
        try:
            job = backend.retrieve_job(jid)
            job.cancel()
            print(f"✔ Cancelled job {jid}")
        except Exception as e:
            print(f"✘ Failed to cancel {jid}: {e}")

    print("\nDone.")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python cancel_jobs_after_index.py <start_index>")
        sys.exit(1)

    idx = int(sys.argv[1])
    cancel_jobs_after_index(idx)
