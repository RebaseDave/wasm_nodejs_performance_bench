import json
from pathlib import Path

import numpy as np
import pandas as pd


def _get(d, *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _num(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _ms_block_stats(obj, prefix, out):
    if not isinstance(obj, dict):
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_p95"] = np.nan
        return
    out[f"{prefix}_mean"] = _num(obj.get("mean", np.nan))
    out[f"{prefix}_p95"] = _num(obj.get("p95", np.nan))


def workload_cat_from_name(workload: str) -> str:
    w = str(workload).lower()
    for k in ("tiny", "small", "medium", "large", "heavy", "ultra"):
        if w.startswith(k + "_"):
            return k
    return "unknown"


def load_all_benchmarks(directory="../../../src/bench/results", pattern="*.json") -> pd.DataFrame:
    rows = []
    for json_file in Path(directory).glob(pattern):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("meta", {}) or {}
        cfg = meta.get("cfg", {}) or {}
        vopts = cfg.get("variantOpts", {}) or {}

        variants = data.get("variants", {}) or {}
        for variant_name, tests in variants.items():
            for test in (tests or []):
                row = {}
                row["source_file"] = json_file.name
                row["timestamp"] = meta.get("timestamp", None)
                row["run_tag"] = meta.get("runTag", None)
                row["node"] = meta.get("node", None)
                row["platform"] = meta.get("platform", None)
                row["arch"] = meta.get("arch", None)
                row["cpu_count"] = _num(_get(meta, "cpus", "count", default=np.nan))
                row["cpu_model"] = _get(meta, "cpus", "model", default=None)
                row["cpu_speed_mhz"] = _num(_get(meta, "cpus", "speed", default=np.nan))
                row["sys_mem_total_mb"] = _num(_get(meta, "memory", "totalMB", default=np.nan))
                row["sys_mem_free_mb"] = _num(_get(meta, "memory", "freeMB", default=np.nan))
                row["sys_mem_used_percent"] = _num(_get(meta, "memory", "usedPercent", default=np.nan))
                row["cfg_mode"] = cfg.get("mode", None)
                row["cfg_iterations"] = _num(cfg.get("iterations", np.nan))
                row["cfg_warmupIters"] = _num(cfg.get("warmupIters", np.nan))
                row["cfg_baseSeed"] = _num(cfg.get("baseSeed", np.nan))
                row["cfg_samplePeakMemory"] = bool(cfg.get("samplePeakMemory", False))
                row["cfg_peakSampleMs"] = _num(cfg.get("peakSampleMs", np.nan))
                row["cfg_cooldownMs"] = _num(cfg.get("cooldownMs", np.nan))
                row["cfg_size"] = _num(vopts.get("size", np.nan))
                row["cfg_enforceTransfer"] = vopts.get("enforceTransfer", None)
                row["variant"] = variant_name
                row["workload"] = test.get("workload", None)
                row["workload_category"] = workload_cat_from_name(row["workload"])
                row["width"] = _num(test.get("width", np.nan))
                row["height"] = _num(test.get("height", np.nan))
                row["kernelSize"] = _num(test.get("kernelSize", np.nan))
                row["iterations"] = _num(test.get("iterations", np.nan))
                row["passes"] = _num(test.get("passes", np.nan))
                
                if np.isfinite(row["width"]) and np.isfinite(row["height"]) and np.isfinite(row["passes"]):
                    row["effectivePixels"] = row["width"] * row["height"] * row["passes"]
                else:
                    row["effectivePixels"] = np.nan
                
                row["mode"] = test.get("mode", None)
                row["concurrency"] = _num(test.get("concurrency", np.nan))
                row["poolSize"] = _num(test.get("poolSize", np.nan))
                row["isWorkerVariant"] = test.get("isWorkerVariant", None)
                
                _ms_block_stats(test.get("latencyMs", None), "latency", row)
                
                lb = test.get("latencyBreakdown", {}) or {}
                _ms_block_stats(lb.get("roundtrip", None), "roundtrip", row)
                _ms_block_stats(lb.get("compute", None), "compute", row)
                _ms_block_stats(lb.get("queueWait", None), "queue_wait", row)
                _ms_block_stats(lb.get("transfer", None), "transfer", row)
                
                row["wall_ms"] = _num(test.get("wallMs", np.nan))
                
                row["cpu_total_micros"] = _num(_get(test, "cpu", "totalMicros", default=np.nan))
                row["cpu_percent"] = _num(_get(test, "cpu", "cpuToWallPercent", default=np.nan))
                
                row["throughput_mpix_sec"] = _num(_get(test, "throughput", "mpixPerSec", default=np.nan))

                el = test.get("eventLoopLag", {}) or {}
                row["el_lag_mean"] = _num(el.get("mean", np.nan))
                row["el_lag_p50"] = _num(el.get("p50", np.nan))
                row["el_lag_p95"] = _num(el.get("p95", np.nan))
                row["el_lag_p99"] = _num(el.get("p99", np.nan))
                row["el_lag_max"] = _num(el.get("max", np.nan))

                def _mb(x):
                    x = _num(x)
                    return x / (1024**2) if np.isfinite(x) else np.nan

                row["mem_peak_rss_mb"] = _mb(_get(test, "memory", "peak", "rss", default=np.nan))
                row["mem_peak_heapUsed_mb"] = _mb(_get(test, "memory", "peak", "heapUsed", default=np.nan))
                
                rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    numeric_cols = [
        "cpu_count", "cpu_speed_mhz",
        "sys_mem_total_mb", "sys_mem_free_mb", "sys_mem_used_percent",
        "cfg_iterations", "cfg_warmupIters", "cfg_baseSeed", "cfg_peakSampleMs", "cfg_cooldownMs", "cfg_size",
        "width", "height", "kernelSize", "iterations", "passes", "effectivePixels",
        "concurrency", "poolSize",
        "latency_mean", "latency_p95",
        "roundtrip_mean", "roundtrip_p95",
        "compute_mean", "compute_p95",
        "queue_wait_mean", "queue_wait_p95",
        "transfer_mean", "transfer_p95",
        "wall_ms", "cpu_total_micros", "cpu_percent",
        "throughput_mpix_sec",
        "el_lag_mean", "el_lag_p50", "el_lag_p95", "el_lag_p99", "el_lag_max",
        "mem_peak_rss_mb", "mem_peak_heapUsed_mb",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["workload_key"] = (
        df["width"].fillna(-1).astype(int).astype(str)
        + "x"
        + df["height"].fillna(-1).astype(int).astype(str)
        + "_k"
        + df["kernelSize"].fillna(-1).astype(int).astype(str)
        + "_p"
        + df["passes"].fillna(-1).astype(int).astype(str)
    )
    df["is_worker"] = (
        df["variant"].astype(str).str.contains("worker", case=False, na=False)
        | (df["isWorkerVariant"] == True)
    )
    df["worker_overhead_ms"] = df["roundtrip_mean"] - df["compute_mean"]
    df["overhead_percent"] = np.where(
        np.isfinite(df["roundtrip_mean"]) & (df["roundtrip_mean"] > 0),
        (df["worker_overhead_ms"] / df["roundtrip_mean"]) * 100.0,
        np.nan,
    )
    df["has_queueing"] = (
        np.isfinite(df["queue_wait_mean"]) & (df["queue_wait_mean"] > 0.01)
    ).astype(int)

    df["queue_ratio"] = np.where(
        np.isfinite(df["roundtrip_mean"]) & (df["roundtrip_mean"] > 0),
        df["queue_wait_mean"] / df["roundtrip_mean"],
        np.nan,
    )
    df["service_ratio"] = np.where(
        np.isfinite(df["roundtrip_mean"]) & (df["roundtrip_mean"] > 0),
        df["compute_mean"] / df["roundtrip_mean"],
        np.nan,
    )
    df["el_lag_severity"] = np.where(
        df["el_lag_p95"] < 20, "healthy",
        np.where(df["el_lag_p95"] < 100, "degraded", "critical")
    )
    df["pixels_per_ms"] = np.where(
        np.isfinite(df["latency_mean"]) & (df["latency_mean"] > 0),
        df["effectivePixels"] / df["latency_mean"],
        np.nan,
    )
    df["rss_mb_per_mpix"] = np.where(
        np.isfinite(df["effectivePixels"]) & (df["effectivePixels"] > 0),
        df["mem_peak_rss_mb"] / (df["effectivePixels"] / 1e6),
        np.nan,
    )
    return df


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_cols = [
        "variant",
        "mode",
        "workload",
        "poolSize",
        "concurrency",
        "cfg_size",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    metric_cols = [
        c
        for c in df.columns
        if (
            c.endswith(("_mean", "_median", "_p50", "_p95", "_p99", "_max"))
            or c
            in (
                "wall_ms",
                "cpu_total_micros",
                "cpu_percent",
                "throughput_mpix_sec",
                "effectivePixels",
                "worker_overhead_ms",
                "overhead_percent",
                "has_queueing",
                "queue_ratio",
                "service_ratio",
                "pixels_per_ms",
                "rss_mb_per_mpix",
                "mem_peak_rss_mb",
                "mem_peak_heapUsed_mb",
            )
        )
    ]
    metric_cols = [c for c in metric_cols if c in df.columns and c not in group_cols]

    g = df.groupby(group_cols, dropna=False)
    mean_df = g[metric_cols].mean(numeric_only=True).reset_index()
    std_df = g[metric_cols].std(ddof=1, numeric_only=True).add_suffix("_std").reset_index()
    out = mean_df.merge(std_df, on=group_cols, how="left")
    out["num_runs"] = g.size().values

    other_cols = [
        "workload_key",
        "workload_category",
        "width",
        "height",
        "kernelSize",
        "passes",
        "iterations",
        "el_lag_severity",
        "isWorkerVariant",
        "is_worker",
        "cfg_mode",
        "cfg_enforceTransfer",
        "cfg_peakSampleMs",
        "cfg_iterations",
    ]
    other_cols = [c for c in other_cols if c in df.columns and c not in group_cols]
    if other_cols:
        first_vals = g[other_cols].first().reset_index()
        out = out.merge(first_vals, on=group_cols, how="left")

    for base in ["latency_mean", "latency_p95", "throughput_mpix_sec", "cpu_percent", "mem_peak_rss_mb"]:
        if base in out.columns and f"{base}_std" in out.columns:
            out[f"{base}_cv"] = np.where(
                np.isfinite(out[base]) & (out[base] != 0),
                out[f"{base}_std"] / out[base],
                np.nan,
            )
    for base in ["latency_mean", "throughput_mpix_sec"]:
        if base in df.columns:
            out[f"{base}_run_min"] = g[base].min().values
            out[f"{base}_run_max"] = g[base].max().values
            out[f"{base}_run_range"] = out[f"{base}_run_max"] - out[f"{base}_run_min"]

    return out


if __name__ == "__main__":
    df_raw = load_all_benchmarks(pattern="*.json")
    df_raw = add_derived_metrics(df_raw)
    df_agg = aggregate_runs(df_raw)
    out_dir = Path(__file__).resolve().parent / "../../processed_data"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(out_dir / "compute_aggregated.csv", index=False, lineterminator='\n')