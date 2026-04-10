import json
import pandas as pd
from pathlib import Path
import numpy as np


def _to_num(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return np.nan
            return float(s)
        return np.nan
    except Exception:
        return np.nan


def _bytes_to_mb(x):
    v = _to_num(x)
    if pd.isna(v):
        return np.nan
    return v / (1024 ** 2)


def load_all_benchmarks(directory="../../../tools/results_system", pattern="*.json"):
    all_results = []

    json_files = list(Path(directory).glob(pattern))

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("meta", {})
        cfg = data.get("cfg", {})
        client = data.get("client", {})
        server = data.get("server", {})

        workload = cfg.get("workload", {})
        load = cfg.get("load", {})

        before = server.get("before", {})
        after = server.get("after", {})

        client_latency = client.get("latency", {}) or {}
        client_latency_client = client_latency.get("client", {}) or {}
        client_latency_server = client_latency.get("server", {}) or {}
        client_latency_compute = client_latency.get("compute", {}) or {}
        client_latency_queuewait = client_latency.get("queueWait", {}) or {}

        client_requests = client.get("requests", {}) or {}

        before_cpu = before.get("cpu", {}) or {}
        after_cpu = after.get("cpu", {}) or {}

        before_mem_current = before.get("memory", {}).get("current", {}) or {}
        before_mem_peak = before.get("memory", {}).get("peak", {}) or {}
        after_mem_current = after.get("memory", {}).get("current", {}) or {}
        after_mem_peak = after.get("memory", {}).get("peak", {}) or {}

        after_requests = after.get("requests", {}) or {}

        before_eventloop_lag = before.get("eventLoopLag", {}) or {}
        after_eventloop_lag = after.get("eventLoopLag", {}) or {}

        after_inflight = after.get("inflight", {}) or {}

        after_queues = after.get("queues", {}) or {}

        after_latency = after.get("latency", {}) or {}
        after_latency_server = after_latency.get("server", {}) or {}
        after_latency_queuewait = after_latency.get("queueWait", {}) or {}
        after_latency_compute = after_latency.get("compute", {}) or {}

        row = {
            "source_file": json_file.name,
            "run_tag": meta.get("tag", "unknown"),
            "rep": _to_num(meta.get("rep", np.nan)),
            "variant": cfg.get("variant", "unknown"),
            "workload": workload.get("name", "unknown"),

            "width": _to_num(workload.get("width", np.nan)),
            "height": _to_num(workload.get("height", np.nan)),
            "kernelSize": _to_num(workload.get("kernelSize", np.nan)),
            "passes": _to_num(workload.get("passes", np.nan)),
            "payloadBytes": _to_num(workload.get("payloadBytes", np.nan)),

            "concurrency": _to_num(load.get("concurrency", np.nan)),
            "warmupSec": _to_num(load.get("warmupSec", np.nan)),
            "durationSec": _to_num(load.get("durationSec", np.nan)),
            "poolSize": _to_num(cfg.get("poolSize", np.nan)),

            "client_latency_mean": _to_num(client_latency_client.get("mean", np.nan)),
            "client_latency_p50": _to_num(client_latency_client.get("p50", np.nan)),
            "client_latency_p95": _to_num(client_latency_client.get("p95", np.nan)),
            "client_latency_p99": _to_num(client_latency_client.get("p99", np.nan)),
            "client_latency_max": _to_num(client_latency_client.get("max", np.nan)),

            "server_latency_mean": _to_num(client_latency_server.get("mean", np.nan)),
            "server_latency_p50": _to_num(client_latency_server.get("p50", np.nan)),
            "server_latency_p95": _to_num(client_latency_server.get("p95", np.nan)),
            "server_latency_p99": _to_num(client_latency_server.get("p99", np.nan)),
            "server_latency_max": _to_num(client_latency_server.get("max", np.nan)),

            "compute_latency_mean": _to_num(client_latency_compute.get("mean", np.nan)),
            "compute_latency_p50": _to_num(client_latency_compute.get("p50", np.nan)),
            "compute_latency_p95": _to_num(client_latency_compute.get("p95", np.nan)),
            "compute_latency_p99": _to_num(client_latency_compute.get("p99", np.nan)),
            "compute_latency_max": _to_num(client_latency_compute.get("max", np.nan)),

            "queuewait_latency_mean": _to_num(client_latency_queuewait.get("mean", np.nan)),
            "queuewait_latency_p50": _to_num(client_latency_queuewait.get("p50", np.nan)),
            "queuewait_latency_p95": _to_num(client_latency_queuewait.get("p95", np.nan)),
            "queuewait_latency_p99": _to_num(client_latency_queuewait.get("p99", np.nan)),
            "queuewait_latency_max": _to_num(client_latency_queuewait.get("max", np.nan)),

            "throughput_rps": _to_num(client.get("throughputRps", np.nan)),

            "client_ok": _to_num(client_requests.get("ok", np.nan)),
            "client_err": _to_num(client_requests.get("err", np.nan)),
            "client_error_rate_percent": _to_num(client_requests.get("errorRatePercent", np.nan)),
            "wall_sec": _to_num(client.get("wallSec", np.nan)),

            "server_requests_total": _to_num(after_requests.get("total", np.nan)),
            "server_requests_ok": _to_num(after_requests.get("ok", np.nan)),
            "server_requests_err": _to_num(after_requests.get("err", np.nan)),
            "server_requests_error_rate_percent": _to_num(after_requests.get("errorRatePercent", np.nan)),
            "server_requests_rps": _to_num(after_requests.get("rps", np.nan)),

            "server_cpu_percent_before": _to_num(before_cpu.get("cpuPercent", np.nan)),
            "server_cpu_percent_after": _to_num(after_cpu.get("cpuPercent", np.nan)),

            "server_eventloop_lag_p95_before": _to_num(before_eventloop_lag.get("p95", np.nan)),
            "server_eventloop_lag_p99_before": _to_num(before_eventloop_lag.get("p99", np.nan)),
            "server_eventloop_lag_max_before": _to_num(before_eventloop_lag.get("max", np.nan)),

            "server_eventloop_lag_p95_after": _to_num(after_eventloop_lag.get("p95", np.nan)),
            "server_eventloop_lag_p99_after": _to_num(after_eventloop_lag.get("p99", np.nan)),
            "server_eventloop_lag_max_after": _to_num(after_eventloop_lag.get("max", np.nan)),

            "server_inflight_current": _to_num(after_inflight.get("current", np.nan)),
            "server_inflight_peak": _to_num(after_inflight.get("peak", np.nan)),

            "server_queue_pending": _to_num(after_queues.get("pending", np.nan)),

            "server_rss_current_before_mb": _bytes_to_mb(before_mem_current.get("rss", np.nan)),
            "server_heap_used_current_before_mb": _bytes_to_mb(before_mem_current.get("heapUsed", np.nan)),

            "server_rss_current_after_mb": _bytes_to_mb(after_mem_current.get("rss", np.nan)),
            "server_heap_used_current_after_mb": _bytes_to_mb(after_mem_current.get("heapUsed", np.nan)),

            "server_rss_peak_before_mb": _bytes_to_mb(before_mem_peak.get("rss", np.nan)),
            "server_heap_used_peak_before_mb": _bytes_to_mb(before_mem_peak.get("heapUsed", np.nan)),

            "server_rss_peak_after_mb": _bytes_to_mb(after_mem_peak.get("rss", np.nan)),
            "server_heap_used_peak_after_mb": _bytes_to_mb(after_mem_peak.get("heapUsed", np.nan)),

            "server_side_latency_mean": _to_num(after_latency_server.get("mean", np.nan)),
            "server_side_latency_p50": _to_num(after_latency_server.get("p50", np.nan)),
            "server_side_latency_p95": _to_num(after_latency_server.get("p95", np.nan)),
            "server_side_latency_p99": _to_num(after_latency_server.get("p99", np.nan)),
            "server_side_latency_max": _to_num(after_latency_server.get("max", np.nan)),

            "server_side_queuewait_mean": _to_num(after_latency_queuewait.get("mean", np.nan)),
            "server_side_queuewait_p50": _to_num(after_latency_queuewait.get("p50", np.nan)),
            "server_side_queuewait_p95": _to_num(after_latency_queuewait.get("p95", np.nan)),
            "server_side_queuewait_p99": _to_num(after_latency_queuewait.get("p99", np.nan)),
            "server_side_queuewait_max": _to_num(after_latency_queuewait.get("max", np.nan)),

            "server_side_compute_mean": _to_num(after_latency_compute.get("mean", np.nan)),
            "server_side_compute_p50": _to_num(after_latency_compute.get("p50", np.nan)),
            "server_side_compute_p95": _to_num(after_latency_compute.get("p95", np.nan)),
            "server_side_compute_p99": _to_num(after_latency_compute.get("p99", np.nan)),
            "server_side_compute_max": _to_num(after_latency_compute.get("max", np.nan)),
        }

        all_results.append(row)

    df = pd.DataFrame(all_results)
    return df


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["poolSize", "payloadBytes", "concurrency", "width", "height", "kernelSize", "passes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "poolSize" in df.columns:
        df["poolSize"] = df["poolSize"].fillna(-1)

    group_cols = [
        "variant", "workload",
        "width", "height", "kernelSize", "passes",
        "poolSize", "concurrency",
        "payloadBytes",
    ]

    metric_cols = [
        "client_latency_mean", "client_latency_p50", "client_latency_p95", "client_latency_p99", "client_latency_max",
        "server_latency_mean", "server_latency_p50", "server_latency_p95", "server_latency_p99", "server_latency_max",
        "compute_latency_mean", "compute_latency_p50", "compute_latency_p95", "compute_latency_p99", "compute_latency_max",
        "queuewait_latency_mean", "queuewait_latency_p50", "queuewait_latency_p95", "queuewait_latency_p99", "queuewait_latency_max",
        "throughput_rps",
        "client_ok", "client_err", "client_error_rate_percent", "wall_sec",
        "server_requests_total", "server_requests_ok", "server_requests_err", 
        "server_requests_error_rate_percent", "server_requests_rps",
        "server_cpu_percent_before", "server_cpu_percent_after",
        "server_eventloop_lag_p95_before", "server_eventloop_lag_p99_before", "server_eventloop_lag_max_before",
        "server_eventloop_lag_p95_after", "server_eventloop_lag_p99_after", "server_eventloop_lag_max_after",
        "server_inflight_current", "server_inflight_peak",
        "server_queue_pending",
        "server_rss_current_before_mb", "server_heap_used_current_before_mb",
        "server_rss_current_after_mb", "server_heap_used_current_after_mb",
        "server_rss_peak_before_mb", "server_heap_used_peak_before_mb",
        "server_rss_peak_after_mb", "server_heap_used_peak_after_mb",
        "server_side_latency_mean", "server_side_latency_p50", "server_side_latency_p95", 
        "server_side_latency_p99", "server_side_latency_max",
        "server_side_queuewait_mean", "server_side_queuewait_p50", "server_side_queuewait_p95",
        "server_side_queuewait_p99", "server_side_queuewait_max",
        "server_side_compute_mean", "server_side_compute_p50", "server_side_compute_p95",
        "server_side_compute_p99", "server_side_compute_max",
    ]

    existing_metrics = [c for c in metric_cols if c in df.columns]
    df_agg = df.groupby(group_cols, as_index=False)[existing_metrics].mean()
    df_agg["num_runs"] = df.groupby(group_cols).size().values
    return df_agg


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "client_latency_mean" in df.columns and "server_latency_mean" in df.columns:
        df["network_overhead_ms"] = (df["client_latency_mean"] - df["server_latency_mean"]).fillna(0)
    else:
        df["network_overhead_ms"] = np.nan

    df["payload_mb"] = df["payloadBytes"] / (1024 ** 2)
    df["payload_mb_per_sec"] = df["throughput_rps"] * df["payload_mb"]

    df["server_rss_delta_mb"] = df["server_rss_peak_after_mb"] - df["server_rss_peak_before_mb"]
    df["server_heap_used_delta_mb"] = df["server_heap_used_peak_after_mb"] - df["server_heap_used_peak_before_mb"]

    if "server_cpu_percent_after" in df.columns and "server_cpu_percent_before" in df.columns:
        df["server_cpu_percent_delta"] = df["server_cpu_percent_after"] - df["server_cpu_percent_before"]
    else:
        df["server_cpu_percent_delta"] = np.nan

    if "server_eventloop_lag_p95_after" in df.columns and "server_eventloop_lag_p95_before" in df.columns:
        df["server_eventloop_lag_p95_delta"] = (
            df["server_eventloop_lag_p95_after"] - df["server_eventloop_lag_p95_before"]
        )
    else:
        df["server_eventloop_lag_p95_delta"] = np.nan

    df["workload_category"] = (
        df["workload"]
        .astype(str)
        .str.lower()
        .str.extract(r"^(small|medium|large|heavy)", expand=False)
        .fillna("unknown")
    )
    df["workload_category"] = pd.Categorical(
        df["workload_category"],
        categories=["small", "medium", "large", "heavy", "unknown"],
        ordered=True
    )
    
    df["is_worker"] = df["variant"].astype(str).str.contains("worker", case=False, na=False)

    if all(col in df.columns for col in ["width", "height", "passes"]):
        df["effectivePixels"] = df["width"] * df["height"] * df["passes"]
    else:
        df["effectivePixels"] = np.nan

    return df


if __name__ == "__main__":
    df_raw = load_all_benchmarks(pattern="*.json")
    df_agg = aggregate_runs(df_raw)
    df_final = add_derived_metrics(df_agg)

    out_dir = (Path(__file__).resolve().parent / "../../processed_data").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / "system_aggregated.csv"
    df_final.to_csv(out, index=False, lineterminator='\n')