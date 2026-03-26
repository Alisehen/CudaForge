#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This module wraps three tasks:
1) Collect core metrics for specified CUDA kernels with Nsight Compute into CSV (`profile_bench`).
2) Extract and clean those metrics into a DataFrame from the CSV (`load_ncu_metrics`).
3) Convert the metrics table into a string suitable for inclusion in an LLM prompt (`metrics_to_prompt`).

Typical usage:
    from gpu_profile_utils import profile_bench, load_ncu_metrics, metrics_to_prompt

    kernel_names = extract_cuda_kernel_names(test_kernel)
    csv_path = profile_bench(kernel_names=kernel_names)
    df = load_ncu_metrics(csv_path, extra_keep=("Kernel Name",))
    prompt_block = metrics_to_prompt(df)
"""

import os
import re
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Sequence, Union, Any
import json, math
import pandas as pd
import numpy as np


__all__ = [
    "METRICS",
    "METRIC_COLUMNS",
    "profile_bench",
    "load_ncu_metrics",
    "metrics_to_prompt",
]

# Keep only the core "kernel performance related" metrics (aligned with `ncu --metrics`)
METRICS = ",".join([
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
])


# List version for convenient header selection
METRIC_COLUMNS: List[str] = [s.strip() for s in METRICS.split(",")]


def _resolve_ncu_bin(
    explicit_ncu_bin: Optional[Union[str, Path]],
    env_path: str,
    conda_bin: Optional[Union[str, Path]],
) -> str:
    candidates: List[Path] = []
    seen: set[str] = set()

    def _add(candidate: Optional[Union[str, Path]]) -> None:
        if not candidate:
            return
        path = Path(candidate).expanduser()
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    env_override = os.environ.get("NCU_BIN") or os.environ.get("NSIGHT_COMPUTE_BIN")
    _add(explicit_ncu_bin)
    _add(env_override)
    if conda_bin:
        _add(Path(conda_bin) / "ncu")

    which_in_env = shutil.which("ncu", path=env_path)
    _add(which_in_env)
    _add(shutil.which("ncu"))
    _add("/usr/local/cuda/bin/ncu")
    _add("/usr/bin/ncu")

    opt_root = Path("/opt/nvidia/nsight-compute")
    if opt_root.exists():
        for match in sorted(opt_root.glob("*/ncu"), reverse=True):
            _add(match)

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    checked = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        "Nsight Compute binary 'ncu' was not found. "
        "Set NCU_BIN or pass --ncu-bin to main.py. "
        f"Checked: {checked}"
    )


def profile_bench(
    bench_py: str = "bench_ref_inputs.py",
    kernel_names: Optional[List[str]] = None,
    conda_bin: Optional[Union[str, Path]] = None,
    out_csv: Union[str, Path] = "ncu_temp.csv",
    repeat: int = 5,
    launch_count: int = 5,
    bench_args: Optional[Sequence[Union[str, Path]]] = None,
    ncu_bin: Optional[Union[str, Path]] = None,
) -> Path:
    csv_path = Path(out_csv).resolve()

    env = os.environ.copy()
    resolved_conda_bin = str(Path(conda_bin).expanduser()) if conda_bin else str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{resolved_conda_bin}:{env.get('PATH', '')}"
    resolved_ncu_bin = _resolve_ncu_bin(ncu_bin, env["PATH"], resolved_conda_bin)
    tmp_ncu_dir = Path.home() / "ncu-tmp"
    tmp_ncu_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp_ncu_dir)
    tmp_ext = tempfile.mkdtemp(prefix="torch_ext_")
    env["TORCH_EXTENSIONS_DIR"] = tmp_ext


    cmd = [
        resolved_ncu_bin,
        "--csv",
        "--page=raw",
        "--kernel-name-base=demangled",
        "--target-processes=all",
        "--replay-mode=kernel",
        "--profile-from-start=on",
        f"--log-file={str(csv_path)}",
        f"--metrics={METRICS}",
        "--launch-skip=0",
        f"--launch-count={max(1, launch_count)}",
        sys.executable, bench_py,
    ]
    if bench_args:
        cmd.extend(str(arg) for arg in bench_args)
    cmd.extend(["--repeat", str(max(1, repeat))])

    # Choose insertion strategy based on number of kernel names
    if kernel_names:
        names = sorted({k.strip() for k in kernel_names if k and k.strip()})
        if names:
            insert_pos = cmd.index(f"--metrics={METRICS}")
            if len(names) == 1:
                # Single name: direct match
                cmd.insert(insert_pos, f"--kernel-name={names[0]}")
            else:
                # Multiple names: merge into a single regex
                pattern = "|".join(re.escape(k) for k in names)
                cmd.insert(insert_pos, f"--kernel-name=::regex:^({pattern})(\\(|$)")

    print("[ncu] running:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(proc.returncode)

    print(f"[ok] CSV written: {csv_path}")
    return csv_path



def load_ncu_metrics(
    csv_path: Union[str, Path] = "ncu_temp.csv",
    columns: Optional[Sequence[str]] = None,
    extra_keep: Optional[Sequence[str]] = ("Kernel Name",),
    coerce_numeric: bool = True,
    name_list: Optional[Sequence[str]] = None,  # New: multiple kernel names
    select: str = "last",                       # Selection policy when multiple rows per name
) -> pd.DataFrame:
    def _contains_unit_token(value: Any, tokens: Sequence[str]) -> bool:
        text = str(value).strip().lower()
        return any(tok in text for tok in tokens)

    def _normalize_kernel_name(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        text = text.split("(", 1)[0].strip()
        text = re.sub(r"<.*?>", "", text)
        if "::" in text:
            text = text.split("::")[-1]
        return text.strip().lower()

    def _kernel_tokens(value: Any) -> set[str]:
        stop_words = {
            "void", "const", "unsigned", "signed", "long", "int", "float",
            "double", "half", "at", "native", "cuda", "launch", "bounds",
        }
        return {
            token for token in re.findall(r"[a-z0-9]+", _normalize_kernel_name(value))
            if token and token not in stop_words
        }

    def _select_rows(matched: pd.DataFrame) -> pd.DataFrame:
        if matched.empty:
            return matched
        if len(matched) == 1:
            return matched
        if select == "first":
            return matched.iloc[[0]]
        if select == "last":
            return matched.iloc[[-1]]
        if select == "max_cycles" and "sm__cycles_active.avg" in matched.columns:
            return matched.sort_values("sm__cycles_active.avg", ascending=False).head(1)
        return matched.iloc[[-1]]

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, comment="=", low_memory=False)

    metric_cols = list(columns) if columns is not None else METRIC_COLUMNS
    keep_cols: List[str] = []
    if extra_keep:
        keep_cols.extend([c for c in extra_keep if c in df.columns])
    keep_cols.extend([c for c in metric_cols if c in df.columns])
    if not keep_cols:
        raise ValueError("No requested columns found in the CSV header.")

    sub = df[keep_cols].copy()

    # Drop the units row
    if len(sub) > 0:
        unit_tokens = ("%", "inst", "cycle", "block", "register", "register/thread")
        first_row = sub.iloc[0].tolist()
        if any(_contains_unit_token(value, unit_tokens) for value in first_row):
            sub = sub.iloc[1:].reset_index(drop=True)

    # Coerce metrics to numeric
    if coerce_numeric:
        metric_in_sub = [c for c in metric_cols if c in sub.columns]
        sub[metric_in_sub] = (
            sub[metric_in_sub]
            .replace({",": "", "%": ""}, regex=True)
            .apply(pd.to_numeric, errors="coerce")
        )

    # ========== Extract by kernel name list ==========
    if name_list:
        if "Kernel Name" not in sub.columns:
            return pd.DataFrame(columns=keep_cols)

        raw_kernel_names = sub["Kernel Name"].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
        normalized_kernel_names = raw_kernel_names.map(_normalize_kernel_name)
        results = []
        for name in name_list:
            if name is None:
                continue
            needle = str(name).strip()
            if not needle:
                continue
            needle_norm = _normalize_kernel_name(needle)

            exact_mask = raw_kernel_names.str.contains(needle, regex=False, na=False)
            if needle_norm:
                exact_mask = exact_mask | normalized_kernel_names.str.contains(
                    needle_norm, regex=False, na=False
                )
            matched = sub[exact_mask]
            if matched.empty:
                continue
            results.append(_select_rows(matched))

        if results:
            sub = pd.concat(results, ignore_index=True)
        else:
            fallback_candidates = []
            seen_candidate_names = set()
            for norm_name in normalized_kernel_names:
                if not norm_name or norm_name in seen_candidate_names:
                    continue
                seen_candidate_names.add(norm_name)
                if "kernel" not in norm_name:
                    continue
                if norm_name.startswith("vectorized_elementwise_kernel"):
                    continue
                matched = sub[normalized_kernel_names == norm_name]
                if matched.empty:
                    continue
                fallback_candidates.append(
                    (norm_name, _kernel_tokens(norm_name), _select_rows(matched))
                )

            fallback_rows = []
            used_candidate_names = set()
            for name in name_list:
                needle_norm = _normalize_kernel_name(name)
                needle_tokens = _kernel_tokens(needle_norm)
                if not needle_tokens:
                    continue

                best_name = None
                best_row = None
                best_score = 0.0
                for candidate_name, candidate_tokens, candidate_row in fallback_candidates:
                    if candidate_name in used_candidate_names:
                        continue
                    overlap = len(needle_tokens & candidate_tokens) / len(needle_tokens)
                    if overlap <= 0.0:
                        continue
                    containment_bonus = 0.25 if (
                        needle_norm in candidate_name or candidate_name in needle_norm
                    ) else 0.0
                    score = overlap + containment_bonus
                    if score > best_score:
                        best_name = candidate_name
                        best_row = candidate_row
                        best_score = score

                if best_name is not None and best_row is not None and best_score >= 0.60:
                    used_candidate_names.add(best_name)
                    fallback_rows.append(best_row)

            if fallback_rows:
                sub = pd.concat(fallback_rows, ignore_index=True)
            else:
                sub = pd.DataFrame(columns=keep_cols)

    return sub


def metrics_to_prompt(
    df: pd.DataFrame,
    title: str = "Here are the GPU profiling metrics:",  # Placeholder, not emitted
    key_by: str = "Kernel Name",
    round_digits: Optional[int] = 3,
    compact: bool = False,
    keep_cols: Optional[List[str]] = None,
) -> str:
    """
    Return **only** the data section as a JSON string:
    {
      "<key>": { "<metric>": <value>, ... }  OR
      "<key>": [{...}, {...}]  # list if there are multiple rows for the same key
    }
    If the key column doesn't exist, return a list of rows: [ {col: val, ...}, ... ]
    """

    def _safe(v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (pd.Timestamp, pd.Timedelta, pd.Interval)):
            return str(v)
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if isinstance(v, float) and round_digits is not None:
            return round(v, round_digits)
        return v

    # Empty table
    if df is None or df.empty:
        return "{}"

    cols = list(df.columns)

    # Round numeric columns
    if round_digits is not None:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df = df.copy()
            df[num_cols] = df[num_cols].round(round_digits)

    # If key column is missing, return a list of rows
    if key_by not in cols:
        rows = [{k: _safe(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]
        return json.dumps(rows, ensure_ascii=False, indent=None if compact else 2)

    # Determine value columns
    value_cols = [c for c in cols if c != key_by]
    if keep_cols is not None:
        value_cols = [c for c in value_cols if c in keep_cols]

    data: Dict[str, Any] = {}
    for rec in df[[key_by] + value_cols].to_dict(orient="records"):
        k = str(rec.pop(key_by))
        val_obj = {ck: _safe(cv) for ck, cv in rec.items()}
        if k in data:
            if isinstance(data[k], list):
                data[k].append(val_obj)
            else:
                data[k] = [data[k], val_obj]
        else:
            data[k] = val_obj

    return json.dumps(data, ensure_ascii=False, indent=None if compact else 2)



if __name__ == "__main__":
    # Simple self-check: doesn't force execution; only runs when this file is executed directly.
    # Note: `profile_bench` requires root privileges and an Nsight Compute environment.
    print("gpu_profile_utils module loaded. Import its functions in your main script.")
