import json
import numpy as np
from typing import Dict, Any, Optional
from typing import Dict, Any, Tuple, Optional, Union    
from collections.abc import Mapping, Sequence
from datetime import datetime
from Analysis_Tools.Extract_Info_From_IonQ_Json import (
    load_json_innerkeys_binary,
    sum_by_bit_condition_dict,
    sum_by_bit_condition_list_no_json,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from collections.abc import Mapping, Sequence


def raw_data_to_fidelity(
    raw_data,
    i,
    j,
    numdata=5,
    numshots=500,
    num_echo=np.array([10, 20]),
    label=None,
    ax=None,
    plot=False,
    save_plot=True,
    save_dir=None,
    save_basename=None,
    save_format="svg",  # or "png"
    grouping: str = "type-major",
):
    """
    Wrapper: raw Forte JSON (dict or list-of-dicts) → pop/par → fidelity fit →
    print + optional plot/save.

    grouping : passed through to data_to_pop_par
        "echo-major" or "type-major" (see data_to_pop_par docstring).
    """

    # -------- Decide mode based on raw_data type --------
    if isinstance(raw_data, Mapping):
        mode = "dict"
    elif isinstance(raw_data, Sequence) and not isinstance(raw_data, (str, bytes)):
        # e.g. list of dicts
        if all(isinstance(el, Mapping) for el in raw_data):
            mode = "list"
        else:
            raise TypeError(
                "raw_data_to_fidelity: sequence input must be a list/tuple of dicts."
            )
    else:
        raise TypeError(
            "raw_data_to_fidelity: raw_data must be a dict or list/tuple of dicts."
        )

    # -------- num_echo handling --------
    if num_echo is None:
        num_echo = np.array([10, 20])
    else:
        num_echo = np.asarray(num_echo, dtype=float)

    # TODO DEBUGGING, comment out later
    # print("TYPE(raw_data):", type(raw_data))
    # print("LEN(raw_data):", len(raw_data) if hasattr(raw_data, "__len__") else None)
    # print("raw_data[0] type:", type(raw_data[0]) if isinstance(raw_data, list) and raw_data else None)
    # print("raw_data[0] keys:", list(raw_data[0].keys())[:5] if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict) else None)
    # print("raw_data[0] sample:", raw_data[0])


    # -------- 1) Extract population & parity from raw data --------
    pop, par, pop_err, par_err = data_to_pop_par(
        raw_data,
        i,
        j,
        numdata=numdata,
        numshots=numshots,
        mode=mode,        # dict vs list
        grouping=grouping # <<< NEW
    )

    # Add an artificial data point for num_echo = 0
    params_at_zero = [0.9837143, 0.96757142, 0.00545, 0.00560]
    pop_with_zero = np.concatenate([[params_at_zero[0]], pop])
    par_with_zero     = np.concatenate([[params_at_zero[1]], par])
    pop_err_with_zero    = np.concatenate([[params_at_zero[2]], pop_err])
    par_err_with_zero    = np.concatenate([[params_at_zero[3]], par_err])
    num_echo_with_zero   = np.concatenate([[0], num_echo])


    # -------- 2) Fit fidelity model + plot --------
    results = analyze_two_qubit_fidelity(
        pop_with_zero,
        par_with_zero,
        pop_err_with_zero,
        par_err_with_zero,
        num_echo=num_echo_with_zero,
        label=label if label is not None else f"Qubits ({i},{j})",
        ax=ax,
        abs_sigma=False,
    )

    # -------- 3) Print fidelity to 6 decimal places --------
    Fg     = results["F_g"]
    Fg_err = results["F_g_err"]
    print(f"Per-gate fidelity F_g: {Fg:.6f} ± {Fg_err:.6f}")
    # -------- 4) Save plot if requested --------
    if save_plot and save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_basename is None: 
            save_basename = f"fidelity_Q{i}_{j}"
        save_path = save_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{save_basename}.{save_format}"
        fig = results["fig"]
        fig.savefig(
            save_path,
            dpi=300,          # good for PNG; harmless for SVG
            bbox_inches="tight",
        )
        print(f"Saved figure to: {save_path}")
    # -------- 5) Show plot if requested --------
    if plot:
        plt.show()
    return results
# -----------------------------------------------------------------------

def analyze_two_qubit_fidelity(
    population,
    parity,
    population_error,
    parity_error,
    num_echo=np.array([10, 20]),
    label="",
    ax=None,
    abs_sigma=True
):
    """
    Fit and plot two-qubit gate fidelity vs number of ZZ gates.

    Parameters
    ----------
    population : array-like
        Measured population data at each num_echo.
    parity : array-like
        Measured parity data at each num_echo.
    population_error : array-like
        1σ uncertainties on population (e.g. from calc_binom_err).
    parity_error : array-like
        1σ uncertainties on parity.
    num_echo : array-like, optional
        Echo counts. Default is [20, 40].
        Number of ZZ gates is computed as 2*num_echo + 1.
    label : str, optional
        Label to show in the plot title.
    ax : matplotlib.axes.Axes, optional
        If provided, plot into this axis. Otherwise a new figure/axis is created.

    Returns
    -------
    results : dict
        {
            "num_gates": array,
            "slope": float,
            "slope_err": float,
            "intercept": float,
            "intercept_err": float,
            "F_g": float,         # per-gate fidelity
            "F_g_err": float,     # uncertainty on per-gate fidelity
            "fig": Figure,
            "ax": Axes
        }
    """
    # Convert to numpy arrays
    population = np.asarray(population, dtype=float)
    parity = np.asarray(parity, dtype=float)
    population_error = np.asarray(population_error, dtype=float)
    parity_error = np.asarray(parity_error, dtype=float)
    num_echo = np.asarray(num_echo, dtype=float)

    # Basic sanity check
    if not (
        population.shape
        == parity.shape
        == population_error.shape
        == parity_error.shape
        == num_echo.shape
    ):
        raise ValueError("population, parity, errors, and num_echo must have the same shape.")

    # Number of ZZ gates
    num_gates = 2 * num_echo + 1

    # Average of population and parity as fidelity proxy
    ydata = 0.5 * (population + parity)
    yerr = 0.5 * (population_error + parity_error)

    # Linear model
    def linear_func(x, a, b):
        return a * x + b

    # Weighted linear fit
    popt, pcov = curve_fit(
        linear_func,
        num_gates,
        ydata,
        sigma=yerr,
        absolute_sigma=abs_sigma,  # yerr treated as true σ
    )
    slope, intercept = popt
    perr = np.sqrt(np.diag(pcov))
    slope_err, intercept_err = perr

    # Per-gate error rate r ≈ -slope, so per-gate fidelity F_g ≈ 1 - r = 1 + slope
    F_g = 1.0 + slope
    F_g_err = slope_err

    # Generate fit line + 1σ band
    x_fit = np.linspace(num_gates.min(), num_gates.max(), 200)
    y_fit = linear_func(x_fit, *popt)

    # Propagate covariance to get uncertainty band
    J = np.vstack((x_fit, np.ones_like(x_fit))).T  # Jacobian wrt [slope, intercept]
    y_var = np.einsum("ij,jk,ik->i", J, pcov, J)
    y_std = np.sqrt(y_var)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    ax.errorbar(
        num_gates,
        population,
        yerr=population_error,
        fmt="o",
        ms=4,
        capsize=3,
        label="population",
    )
    ax.errorbar(
        num_gates,
        parity,
        yerr=parity_error,
        fmt="s",
        ms=4,
        capsize=3,
        label="parity",
    )
    ax.errorbar(
        num_gates,
        ydata,
        yerr=yerr,
        fmt="d",
        ms=4,
        capsize=3,
        label="fidelity",
    )

    ax.plot(x_fit, y_fit, "-", label="Fit")
    ax.fill_between(x_fit, y_fit - y_std, y_fit + y_std, alpha=0.3, label="Fit ±1σ")

    title_prefix = f"{label}: " if label else ""
    # 1 - F_g per gate ≈ -slope
    ax.set_title(
        f"{title_prefix}1 - F$_g$ ≈ {-slope:.4f} ± {slope_err:.4f}",
        fontsize=10,
    )
    ax.set_xlabel("Number of ZZ gates")
    ax.set_ylabel("Fidelity")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return {
        "num_gates": num_gates,
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "intercept_err": intercept_err,
        "F_g": F_g,
        "F_g_err": F_g_err,
        "fig": fig,
        "ax": ax,
    }


# ---------------------------------------

def calc_binom_err(population, numshots):
    population = np.abs(population)
    binom_err = np.sqrt(population * (1 - population) / numshots)
    # Convert to array, then replace zeros safely
    binom_err = np.where(binom_err == 0, 0.01, binom_err)
    return binom_err

def calc_parity_err(parity, numshots):
    binom_err = np.sqrt((1 + parity) * (1 - parity) / numshots)
    binom_err = np.where(binom_err == 0, 0.01, binom_err)
    return binom_err

# -----------------------------------------------------------------------

def data_to_pop_par(data, q_i, q_j, numdata, numshots, mode=None, grouping=None, print_data=False):
    """
    Compute population and parity vs echo from either raw bitstring data
    OR from precomputed P00, P01, P10, P11 lists.

    Raw data keys may look like: "010010... 01"
    We parse the reduced 2-bit outcome as: key.split()[-1]
    (matching aggregate_two_qubit_probabilities).
    """

    # ---------- 0) Check if `data` is already P00/P01/P10/P11 ----------
    if (
        isinstance(data, Mapping)
        and {"P00", "P01", "P10", "P11"}.issubset(data.keys())
    ):
        # Treat as precomputed probabilities
        P00 = np.asarray(data["P00"], dtype=float)
        P01 = np.asarray(data["P01"], dtype=float)
        P10 = np.asarray(data["P10"], dtype=float)
        P11 = np.asarray(data["P11"], dtype=float)

        n0, n1, n2, n3 = map(len, (P00, P01, P10, P11))
        if not (n0 == n1 == n2 == n3):
            raise ValueError(
                "data_to_pop_par: P00, P01, P10, P11 must all have the same length."
            )

    else:
        # ---------- 1) Normalize to list of per-run mappings (raw data path) ----------
        if isinstance(data, Mapping):
            # If dict-of-runs, keep only the per-run mappings
            runs = [m for _, m in data.items()]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            runs = list(data)
        else:
            raise TypeError(
                "data_to_pop_par: data must be a dict, list/tuple of dicts, "
                "or a dict with keys 'P00','P01','P10','P11'."
            )

        if not runs:
            raise ValueError("data_to_pop_par: empty data.")

        # ---------- 2) Build P00, P01, P10, P11 per run from 2-bit reduced key ----------
        P00_list, P01_list, P10_list, P11_list = [], [], [], []

        allowed = {"00", "01", "10", "11"}

        for mapping in runs:
            if not isinstance(mapping, Mapping):
                raise TypeError(
                    "data_to_pop_par: each run must be a mapping of bitstring -> counts."
                )

            grouped_counts = {"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0}

            # IMPORTANT: match aggregate_two_qubit_probabilities()
            total_counts = 0.0
            for key, count in mapping.items():
                if not isinstance(count, (int, float)):
                    continue
                total_counts += float(count)

                if not isinstance(key, str):
                    continue

                suffix = key.split()[-1]  # <-- exact same parse rule
                if suffix in allowed:
                    grouped_counts[suffix] += float(count)

            if total_counts == 0:
                # Keep behavior consistent with your existing code path (no NaNs)
                P00_list.append(0.0)
                P01_list.append(0.0)
                P10_list.append(0.0)
                P11_list.append(0.0)
            else:
                P00_list.append(grouped_counts["00"] / total_counts)
                P01_list.append(grouped_counts["01"] / total_counts)
                P10_list.append(grouped_counts["10"] / total_counts)
                P11_list.append(grouped_counts["11"] / total_counts)

        P00 = np.array(P00_list, dtype=float)
        P01 = np.array(P01_list, dtype=float)
        P10 = np.array(P10_list, dtype=float)
        P11 = np.array(P11_list, dtype=float)

    # ---------- 3) From P00..P11 to per-circuit population & parity ----------
    pop = P00 + P11
    par = P00 + P11 - P01 - P10

    if print_data:
        print("population (per circuit):", pop)
        print("parity (per circuit):    ", par)

    # ---------- 4) Group into blocks of size `numdata` ----------
    population = pop[::numdata]

    if numdata == 5:
        parity = 0.25 * (
            par[1::numdata] - par[2::numdata] + par[3::numdata] - par[4::numdata]
        )
        par_err = calc_parity_err(parity, numshots)
        pop_err = calc_binom_err(population, numshots)

    elif numdata == 3:
        parity = 0.5 * (par[1::numdata] - par[2::numdata])
        par_err = calc_parity_err(parity, numshots)
        pop_err = calc_binom_err(population, numshots)

    else:
        raise ValueError("data_to_pop_par: numdata must be 3 or 5.")

    print("population (grouped):", population)
    print("parity (grouped):    ", parity)

    return population, parity, pop_err, par_err
