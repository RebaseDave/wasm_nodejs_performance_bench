import os
import pandas as pd
import numpy as np


COMPUTE_CSV = "../../processed_data/compute_aggregated.csv"
OUTPUT_DIR  = "../../results/tables/compute/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANT_ORDER = [
    "js_pure", "js_worker",
    "js_wasm", "js_worker_wasm",
    "js_wasm_simd", "js_worker_wasm_simd",
]

WORKLOAD_ORDER = [
    "tiny_128x128_k3_p1",
    "small_512x512_k5_p1",
    "medium_1920x1080_k15_p1",
    "large_1920x1080_k15_p10",
    "heavy_2560x1440_k31_p5",
    "ultra_3840x2160_k31_p5",
]

EFFECTIVE_PIX_MAP = {
    "tiny_128x128_k3_p1":       16_384,
    "small_512x512_k5_p1":     262_144,
    "medium_1920x1080_k15_p1": 2_073_600,
    "large_1920x1080_k15_p10": 20_736_000,
    "heavy_2560x1440_k31_p5":  18_432_000,
    "ultra_3840x2160_k31_p5":  41_472_000,
}

def sort_variants(df, col="variant"):
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=VARIANT_ORDER, ordered=True)
    return df.sort_values(col)

def sort_workloads(df, col="workload"):
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=WORKLOAD_ORDER, ordered=True)
    return df.sort_values(col)

df = pd.read_csv(COMPUTE_CSV)

df["poolSize_int"] = df["poolSize"].astype(int)
df["concurrency_int"] = df["concurrency"].astype(int)

df["effectivePixels_mpix"] = (df["effectivePixels"] / 1e6).round(3)

single = df[(df["mode"] == "single") & (df["concurrency_int"] == 1)].copy()
single_clean = single.copy()

t01 = (
    single_clean
    .groupby(["variant", "workload"], observed=True)
    .agg(
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        compute_mean_ms=("compute_mean", "mean"),
        compute_p95_ms=("compute_p95", "mean"),
        latency_mean_ms=("latency_mean", "mean"),
        latency_p95_ms=("latency_p95", "mean"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
        throughput_std=("throughput_mpix_sec_std", "mean"),
        cpu_percent=("cpu_percent", "mean"),
        mem_peak_rss_mb=("mem_peak_rss_mb", "mean"),
        el_lag_p95_ms=("el_lag_p95", "mean"),
        num_runs=("num_runs", "first"),
    )
    .reset_index()
    .round(3)
)
t01 = sort_variants(sort_workloads(t01))
t01.to_csv(os.path.join(OUTPUT_DIR, "01_single_mode_overview.csv"), index=False)

t02 = (
    single_clean
    .groupby(["variant", "workload"], observed=True)
    .agg(
        width=("width", "first"),
        height=("height", "first"),
        kernelSize=("kernelSize", "first"),
        passes=("passes", "first"),
        effectivePixels=("effectivePixels", "first"),
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
        throughput_std=("throughput_mpix_sec_std", "mean"),
        compute_mean_ms=("compute_mean", "mean"),
        latency_mean_ms=("latency_mean", "mean"),
    )
    .reset_index()
    .round(3)
)
t02 = sort_variants(sort_workloads(t02))
t02.to_csv(os.path.join(OUTPUT_DIR, "02_throughput_scaling_by_workload.csv"), index=False)

base_t03 = (
    single_clean
    .groupby(["variant", "workload"], observed=True)["throughput_mpix_sec"]
    .mean()
    .reset_index()
)

ref = (
    base_t03[base_t03["variant"] == "js_pure"]
    .rename(columns={"throughput_mpix_sec": "ref_throughput"})
    .drop(columns=["variant"])
)

t03 = (
    base_t03
    .merge(ref, on="workload")
    .assign(
        speedup=lambda x: (x["throughput_mpix_sec"] / x["ref_throughput"]).round(3),
        throughput_mpix_sec=lambda x: x["throughput_mpix_sec"].round(3),
        ref_throughput=lambda x: x["ref_throughput"].round(3),
    )
    .rename(columns={
        "throughput_mpix_sec": "variant_throughput_mpix_sec",
        "ref_throughput": "js_pure_throughput_mpix_sec",
    })
)
t03 = sort_variants(sort_workloads(t03))
t03.to_csv(os.path.join(OUTPUT_DIR, "03_wasm_vs_js_speedup.csv"), index=False)

workers_single = single_clean[single_clean["is_worker"] == True].copy()

t04 = (
    workers_single
    .groupby(["variant", "workload"], observed=True)
    .agg(
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        worker_overhead_ms=("worker_overhead_ms", "mean"),
        overhead_percent=("overhead_percent", "mean"),
        compute_mean_ms=("compute_mean", "mean"),
        latency_mean_ms=("latency_mean", "mean"),
        el_lag_p95_ms=("el_lag_p95", "mean"),
        el_lag_max_ms=("el_lag_max", "mean"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
    )
    .reset_index()
    .round(3)
)
t04 = sort_variants(sort_workloads(t04))
t04.to_csv(os.path.join(OUTPUT_DIR, "04_worker_overhead.csv"), index=False)

load = df[df["mode"] == "load"].copy()

workers_load = load[
    (load["is_worker"] == True) &
    (load["poolSize_int"] == load["concurrency_int"])
].copy()

nonworker_load = load[load["is_worker"] == False].copy()
nonworker_load = nonworker_load[nonworker_load["poolSize_int"] == 1].copy()

combined_load = pd.concat([workers_load, nonworker_load], ignore_index=True)

t05 = (
    combined_load
    .groupby(["variant", "workload", "concurrency_int"], observed=True)
    .agg(
        poolSize=("poolSize_int", "first"),
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
        throughput_std=("throughput_mpix_sec_std", "mean"),
        latency_mean_ms=("latency_mean", "mean"),
        latency_p95_ms=("latency_p95", "mean"),
        compute_mean_ms=("compute_mean", "mean"),
        cpu_percent=("cpu_percent", "mean"),
        el_lag_p95_ms=("el_lag_p95", "mean"),
        el_lag_severity=("el_lag_severity", "first"),
        mem_peak_rss_mb=("mem_peak_rss_mb", "mean"),
        num_runs=("num_runs", "first"),
    )
    .reset_index()
    .rename(columns={"concurrency_int": "concurrency"})
    .round(3)
)
t05 = sort_variants(sort_workloads(t05))
t05.to_csv(os.path.join(OUTPUT_DIR, "05_load_mode_concurrency.csv"), index=False)

base_t06 = t02[["variant", "workload", "effectivePixels", "effectivePixels_mpix",
                  "throughput_mpix_sec", "compute_mean_ms"]].copy()

ref_t06 = (
    base_t06[base_t06["variant"] == "js_pure"]
    [["workload", "throughput_mpix_sec", "compute_mean_ms"]]
    .rename(columns={
        "throughput_mpix_sec": "js_pure_throughput",
        "compute_mean_ms": "js_pure_compute_ms",
    })
)

t06 = (
    base_t06
    .merge(ref_t06, on="workload")
    .assign(
        speedup_vs_jspure=lambda x: (x["throughput_mpix_sec"] / x["js_pure_throughput"]).round(3),
        compute_delta_ms=lambda x: (x["compute_mean_ms"] - x["js_pure_compute_ms"]).round(3),
        wasm_faster=lambda x: x["speedup_vs_jspure"] > 1.0,
    )
    .drop(columns=["js_pure_throughput", "js_pure_compute_ms"])
)

t06["pixels_M"] = (t06["effectivePixels"] / 1e6).round(2)

t06 = sort_variants(sort_workloads(t06))
t06.to_csv(os.path.join(OUTPUT_DIR, "06_breakeven_analysis.csv"), index=False)

el_single = (
    single_clean
    .groupby(["variant", "workload"], observed=True)
    .agg(
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        el_lag_p95_ms=("el_lag_p95", "mean"),
        el_lag_p99_ms=("el_lag_p99", "mean"),
        el_lag_max_ms=("el_lag_max", "mean"),
        cpu_percent=("cpu_percent", "mean"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
    )
    .reset_index()
    .assign(mode="single", concurrency=1)
    .round(4)
)

load_c1 = load[load["concurrency_int"] == 1].copy()
load_c1_nonw = load_c1[(load_c1["is_worker"] == False) & (load_c1["poolSize_int"] == 1)]
load_c1_w = load_c1[(load_c1["is_worker"] == True) & (load_c1["poolSize_int"] == 1)]
load_c1_combined = pd.concat([load_c1_nonw, load_c1_w], ignore_index=True)

el_load = (
    load_c1_combined
    .groupby(["variant", "workload"], observed=True)
    .agg(
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        el_lag_p95_ms=("el_lag_p95", "mean"),
        el_lag_p99_ms=("el_lag_p99", "mean"),
        el_lag_max_ms=("el_lag_max", "mean"),
        cpu_percent=("cpu_percent", "mean"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
    )
    .reset_index()
    .assign(mode="load_c1", concurrency=1)
    .round(4)
)

t07 = pd.concat([el_single, el_load], ignore_index=True)
t07 = sort_variants(sort_workloads(t07))
t07.to_csv(os.path.join(OUTPUT_DIR, "07_event_loop_and_cpu.csv"), index=False)

t08 = (
    single_clean
    .groupby(["variant", "workload"], observed=True)
    .agg(
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        latency_mean_cv=("latency_mean_cv", "mean"),
        throughput_cv=("throughput_mpix_sec_cv", "mean"),
        latency_run_min=("latency_mean_run_min", "mean"),
        latency_run_max=("latency_mean_run_max", "mean"),
        latency_run_range=("latency_mean_run_range", "mean"),
        throughput_run_min=("throughput_mpix_sec_run_min", "mean"),
        throughput_run_max=("throughput_mpix_sec_run_max", "mean"),
        throughput_run_range=("throughput_mpix_sec_run_range", "mean"),
        num_runs=("num_runs", "first"),
    )
    .reset_index()
    .round(4)
)
t08 = sort_variants(sort_workloads(t08))
t08.to_csv(os.path.join(OUTPUT_DIR, "08_stability_cv_and_runrange.csv"), index=False)

t09 = (
    single_clean
    .groupby(["variant", "workload"], observed=True)
    .agg(
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        mem_peak_rss_mb=("mem_peak_rss_mb", "mean"),
        mem_peak_rss_std=("mem_peak_rss_mb_std", "mean"),
        mem_peak_heap_mb=("mem_peak_heapUsed_mb", "mean"),
        mem_peak_heap_std=("mem_peak_heapUsed_mb_std", "mean"),
        rss_mb_per_mpix=("rss_mb_per_mpix", "mean"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
    )
    .reset_index()
    .round(3)
)
t09 = sort_variants(sort_workloads(t09))
t09.to_csv(os.path.join(OUTPUT_DIR, "09_memory_usage.csv"), index=False)

load = df[df["mode"] == "load"].copy()

pure_load = (
    load[(load["is_worker"] == False) & (load["poolSize_int"] == 1)]
    .groupby(["variant", "workload"], observed=True)
    .agg(
        concurrency=("concurrency_int", "first"),
        poolSize=("poolSize_int", "first"),
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
        throughput_std=("throughput_mpix_sec_std", "mean"),
        latency_mean_ms=("latency_mean", "mean"),
        latency_p95_ms=("latency_p95", "mean"),
        compute_mean_ms=("compute_mean", "mean"),
        cpu_percent=("cpu_percent", "mean"),
        el_lag_p95_ms=("el_lag_p95", "mean"),
        el_lag_max_ms=("el_lag_max", "mean"),
        el_lag_severity=("el_lag_severity", "first"),
        mem_peak_rss_mb=("mem_peak_rss_mb", "mean"),
        worker_overhead_ms=("worker_overhead_ms", "mean"),
        num_runs=("num_runs", "first"),
    )
    .reset_index()
)

worker_load = (
    load[
        (load["is_worker"] == True) &
        (load["poolSize_int"] == load["concurrency_int"])
    ]
    .groupby(["variant", "workload", "concurrency_int"], observed=True)
    .agg(
        poolSize=("poolSize_int", "first"),
        effectivePixels_mpix=("effectivePixels_mpix", "first"),
        throughput_mpix_sec=("throughput_mpix_sec", "mean"),
        throughput_std=("throughput_mpix_sec_std", "mean"),
        latency_mean_ms=("latency_mean", "mean"),
        latency_p95_ms=("latency_p95", "mean"),
        compute_mean_ms=("compute_mean", "mean"),
        cpu_percent=("cpu_percent", "mean"),
        el_lag_p95_ms=("el_lag_p95", "mean"),
        el_lag_max_ms=("el_lag_max", "mean"),
        el_lag_severity=("el_lag_severity", "first"),
        mem_peak_rss_mb=("mem_peak_rss_mb", "mean"),
        worker_overhead_ms=("worker_overhead_ms", "mean"),
        num_runs=("num_runs", "first"),
    )
    .reset_index()
    .rename(columns={"concurrency_int": "concurrency"})
)

t10 = pd.concat([pure_load, worker_load], ignore_index=True)

ref_worker = (
    worker_load[worker_load["variant"] == "js_worker"]
    [["workload", "concurrency", "throughput_mpix_sec"]]
    .rename(columns={"throughput_mpix_sec": "js_worker_throughput"})
)
t10 = t10.merge(ref_worker, on=["workload", "concurrency"], how="left")
t10["speedup_vs_js_worker"] = (
    t10["throughput_mpix_sec"] / t10["js_worker_throughput"]
).round(3)
t10.loc[t10["variant"] == "js_pure", "speedup_vs_js_worker"] = float("nan")
t10.drop(columns=["js_worker_throughput"], inplace=True)

numeric_t10 = [
    "throughput_mpix_sec", "throughput_std",
    "latency_mean_ms", "latency_p95_ms", "compute_mean_ms",
    "cpu_percent", "el_lag_p95_ms", "el_lag_max_ms",
    "mem_peak_rss_mb", "worker_overhead_ms",
]
t10[numeric_t10] = t10[numeric_t10].round(3)

t10["variant"] = pd.Categorical(t10["variant"], categories=VARIANT_ORDER, ordered=True)
t10["workload"] = pd.Categorical(t10["workload"], categories=WORKLOAD_ORDER, ordered=True)
t10 = t10.sort_values(["workload", "concurrency", "variant"]).reset_index(drop=True)

t10 = t10[[
    "variant", "workload", "concurrency", "poolSize",
    "effectivePixels_mpix",
    "throughput_mpix_sec", "throughput_std", "speedup_vs_js_worker",
    "latency_mean_ms", "latency_p95_ms",
    "compute_mean_ms",
    "cpu_percent",
    "el_lag_p95_ms", "el_lag_max_ms", "el_lag_severity",
    "mem_peak_rss_mb",
    "worker_overhead_ms",
    "num_runs",
]]
t10.to_csv(os.path.join(OUTPUT_DIR, "10_load_mode_overview.csv"), index=False)