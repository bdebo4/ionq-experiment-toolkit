import json
import numpy as np
from typing import Dict, Any, Optional
from typing import Any, Dict, Optional, Union, List
from collections.abc import Mapping, Sequence
from pathlib import Path


def load_json_innerkeys_binary(
    json_file: Union[str, Path],
    num_keys: Optional[int] = None,
    bit_width: int = 36,
    coerce_value: bool = True,
):
    """
    Load an IonQ-style JSON file and return data where the *inner* keys
    are binary bitstrings.

    Supported input layouts
    -----------------------
    1. Dict → multi-run or multi-experiment:
       {
           "run_000": {"0": 10, "1": 5, ...},      # integer index keys
           "run_001": {"000...101": 7, ...},       # OR direct bitstrings
           ...
       }

       or a single histogram:
       {
           "0": 10,
           "1": 5,
           ...
       }

       In the single-histogram case, we wrap as:
           { "run_0": { ... } }

       Return type: dict[str, dict[str, float]]

    2. List of dicts (e.g. cumulative data already in bitstring form):
       [
           {"000...000": 260, "000...1000": 2, ...},
           {"000...000": 255, ...},
           ...
       ]

       Return type: list[dict[str, float]]

    In all cases, keys in the *inner* dicts are converted to fixed-width
    binary strings when they are integer indices, or normalized when they
    are already bitstrings.

    Parameters
    ----------
    json_file : str or Path
        Path to the JSON file.
    num_keys : Optional[int]
        For dict input: if provided and the JSON is multi-run, only the
        first `num_keys` top-level keys are processed. Ignored for list
        input and for single-histogram dict input.
    bit_width : int
        Number of bits for binary formatting (e.g., 10 or 36).
    coerce_value : bool
        Whether to convert values to float (default True).

    Returns
    -------
    dict[str, dict[str, Any]] or list[dict[str, Any]]
        - dict case: { top_key: { "<bit_width>-bit-binary": value, ... }, ... }
        - list case: [ { "<bit_width>-bit-binary": value, ... }, ... ]
    """

    json_file = Path(json_file)
    with open(json_file, "r") as f:
        raw = json.load(f)

    # Helper: convert key to normalized bitstring
    def _to_bits(key: Any) -> Optional[str]:
        # Case 1: integer or integer-like string → treat as index
        try:
            if isinstance(key, (int, str)) and str(key).strip().isdigit():
                i = int(key)
                if i < 0:
                    return None
                return format(i, f"0{bit_width}b")
        except (TypeError, ValueError):
            pass

        if isinstance(key, str):
            s = key.strip()
            if not s:
                return None

            # handle keys like "<36bits> <2bits>"
            parts = s.split()
            if len(parts) == 2 and set(parts[0]).issubset({"0","1"}) and set(parts[1]).issubset({"0","1"}):
                full_bits, reduced_bits = parts[0], parts[1]

                # normalize full_bits to bit_width
                if len(full_bits) < bit_width:
                    full_bits = full_bits.zfill(bit_width)
                elif len(full_bits) > bit_width:
                    full_bits = full_bits[-bit_width:]

                # keep the reduced suffix exactly as-is (typically 2 bits)
                return f"{full_bits} {reduced_bits}"

            # original Case 2: pure bitstring already
            if set(s).issubset({"0", "1"}):
                if len(s) < bit_width:
                    return s.zfill(bit_width)
                elif len(s) > bit_width:
                    return s[-bit_width:]
                else:
                    return s

        return None


    # ------------------------------------------------------------------
    # Case A: dict top-level
    # ------------------------------------------------------------------
    if isinstance(raw, Mapping):
        values = list(raw.values())
        is_multi_run = bool(values) and all(isinstance(v, Mapping) for v in values)

        if is_multi_run:
            # Multi-run / multi-experiment dict
            run_dict: Dict[str, Dict[str, Any]] = dict(raw)  # shallow copy
            top_keys = list(run_dict.keys()) if num_keys is None else list(run_dict.keys())[:num_keys]
        else:
            # Single histogram dict → wrap as a single run
            run_dict = {"run_0": raw}
            top_keys = ["run_0"]

        out: Dict[str, Dict[str, Any]] = {}

        for tk in top_keys:
            inner = run_dict[tk]
            if not isinstance(inner, Mapping):
                continue

            converted: Dict[str, Any] = {}
            for k, v in inner.items():
                b = _to_bits(k)
                if b is None:
                    continue
                converted[b] = float(v) if coerce_value else v

            out[tk] = converted

        return out

    # ------------------------------------------------------------------
    # Case B: list top-level (e.g. list-of-dicts cumulative data)
    # ------------------------------------------------------------------
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out_list: List[Dict[str, Any]] = []

        for elem in raw:
            if not isinstance(elem, Mapping):
                # Skip non-dict elements defensively
                continue
            converted: Dict[str, Any] = {}
            for k, v in elem.items():
                b = _to_bits(k)
                if b is None:
                    continue
                converted[b] = float(v) if coerce_value else v
            out_list.append(converted)

        return out_list

    # ------------------------------------------------------------------
    # Anything else is unsupported
    # ------------------------------------------------------------------
    raise TypeError(
        f"Unsupported JSON top-level type: {type(raw)}. "
        "Expected dict or list of dicts."
   )

# def load_json_innerkeys_binary(
#     json_file: str,
#     num_keys: Optional[int] = None,
#     bit_width: int = 36,
#     coerce_value: bool = True,
# ) -> Dict[str, Dict[str, Any]]:
#     """
#     Load an IonQ-style JSON histogram and return a sparse mapping whose inner
#     keys are *binary* bitstrings.

#     Supported layouts
#     -----------------
#     1. Multi-run / multi-experiment (36-qubit or 2-qubit cumulative, etc.)
#        Example:
#        {
#            "run_000": {"0": 0.1, "1": 0.2, ...},
#            "run_001": {"0": 0.05, ...},
#            ...
#        }
#        or
#        {
#            "<experiment_id_1>": {"0": 0.1, "2148007936": 0.4, ...},
#            "<experiment_id_2>": {...},
#            ...
#        }

#     2. Single histogram (e.g. 10-qubit Forte job)
#        Example:
#        {
#            "0": 0.1,
#            "1": 0.2,
#            ...
#        }
#        In this case the file is wrapped as a single entry "run_0".

#     Parameters
#     ----------
#     json_file : str
#         Path to the JSON file.
#     num_keys : Optional[int]
#         If provided and the JSON is multi-run, only the first `num_keys`
#         top-level keys are processed. Ignored for single-histogram files.
#     bit_width : int
#         Number of bits for binary formatting (e.g. 10 or 36).
#         For Forte-style bitfields (including 2-qubit cumulative data),
#         you typically want 36 here.
#     coerce_value : bool
#         Whether to convert values to float (default True).

#     Returns
#     -------
#     Dict[str, Dict[str, Any]]
#         For multi-run input:
#             { top_key: { "<bit_width>-bit-binary(index)": value, ... }, ... }
#         For single-histogram input:
#             { "run_0": { "<bit_width>-bit-binary(index)": value, ... } }
#     """
#     with open(json_file, "r") as f:
#         raw = json.load(f)

#     if not isinstance(raw, dict):
#         raise TypeError(f"Expected JSON top level to be a dict, got {type(raw)}")

#     # Detect layout: multi-run (dict of dicts) vs single histogram (dict of scalars)
#     values = list(raw.values())
#     is_multi_run = bool(values) and all(isinstance(v, dict) for v in values)

#     if is_multi_run:
#         # 36-qubit / 2-qubit cumulative / multi-experiment style
#         run_dict: Dict[str, Dict[str, Any]] = raw  # type: ignore[assignment]
#         top_keys = list(run_dict.keys()) if num_keys is None else list(run_dict.keys())[:num_keys]
#     else:
#         # Single histogram (e.g. 10-qubit example)
#         run_dict = {"run_0": raw}
#         top_keys = ["run_0"]

#     out: Dict[str, Dict[str, Any]] = {}

#     for tk in top_keys:
#         inner = run_dict[tk]
#         if not isinstance(inner, dict):
#             # Defensive: skip malformed entries
#             continue

#         sparse_map: Dict[str, Any] = {}

#         for idx_str, val in inner.items():
#             # IonQ JSON uses indices as stringified integers
#             try:
#                 i = int(idx_str)
#                 if i < 0:
#                     continue
#             except (TypeError, ValueError):
#                 # Skip non-integer keys (metadata, etc.)
#                 continue

#             # Fixed-width binary string (e.g., 10 or 36 bits).
#             # If `bit_width` is less than the bits needed to represent `i`,
#             # Python will still return the full bitstring (no truncation),
#             # so you should set `bit_width` to the device width you care about.
#             b = format(i, f"0{bit_width}b")
#             sparse_map[b] = float(val) if coerce_value else val

#         out[tk] = sparse_map

#     return out

def sum_by_bit_condition_dict(
    binary_dict: dict[str, dict[str, float]],
    a: int,
    b: int,
    value_a: int = 0,
    value_b: int = 0
) -> np.ndarray:
    """
    For each top-level key, sum the *values* whose binary index
    has bit[a] == bit[b] == value.

    Works directly on the sparse dictionary, avoiding huge zero arrays.

    Parameters
    ----------
    binary_dict : dict
        { top_key: { binary_str (e.g. '000...101'): numeric_value, ... } }
    a, b : int
        Bit positions (0 = least significant bit, rightmost)
    value : int
        Bit value to match (0 or 1)

    Returns
    -------
    np.ndarray
        Array of sums, one element per top-level key, in insertion order.
    """
    if value_a not in (0, 1) or value_b not in (0,1):
        raise ValueError("value must be 0 or 1")

    bit_char_a = str(value_a)
    bit_char_b = str(value_b)
    results = []

    for top_key, mapping in binary_dict.items():
        total = 0.0
        for bin_str, val in mapping.items():
            # Compare bits from rightmost side
            if bin_str[-1 - a] == bit_char_a and bin_str[-1 - b] == bit_char_b:
                total += val
        results.append(total)

    return np.array(results)


def sum_by_bit_condition_list_no_json(
    binary_data,
    a: int,
    b: int,
    value_a: int = 0,
    value_b: int = 0,
    shots: int = 100
) -> np.ndarray:
    """
    For each dict in `binary_list`, sum the values whose binary index (string)
    satisfies: bit[a] == value_a AND bit[b] == value_b.
    Positions are counted from the *right* (LSB) with 0 = rightmost bit.
    Parameters
    ----------
    binary_list : sequence of dicts
        Each element: { "<binary_str>": numeric_value, ... }
    a, b : int
        Bit positions (0 = least significant bit, rightmost).
    value_a, value_b : int
        Bit values to match (0 or 1).
    Returns
    -------
    np.ndarray
        One sum per dict in `binary_list`, in order.
    """
        # ---- Normalize input ----
    if isinstance(binary_data, Mapping):
        # User passed a single dict
        binary_list = [binary_data]
    elif isinstance(binary_data, Sequence) and not isinstance(binary_data, (str, bytes)):
        # Already a list/tuple
        binary_list = list(binary_data)
    else:
        raise TypeError("Input must be a dict or list/tuple of dicts.")
    #-----------------------------------------------------------------------
    if value_a not in (0, 1) or value_b not in (0, 1):
        raise ValueError("value_a and value_b must be 0 or 1")
    bit_a = str(value_a)
    bit_b = str(value_b)
    out = []
    for mapping in binary_list:
        total = 0.0
        for bin_str, val in mapping.items():
            # Treat missing left bits as 0 if strings have varying widths.
            # Index from right: -1-a and -1-b.
            def safe_bit(s: str, pos_from_right: int) -> str:
                idx = -1 - pos_from_right
                return s[idx] if -idx <= len(s) else "0"
            if safe_bit(bin_str, a) == bit_a and safe_bit(bin_str, b) == bit_b:
                total += float(val)/shots
        out.append(total)
    return np.array(out)

