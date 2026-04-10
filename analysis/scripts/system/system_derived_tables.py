import os
import pandas as pd
import numpy as np

SYSTEM_CSV  = "../../processed_data/system_aggregated.csv"
DERIVED_DIR = "../../results/tables/system/"

os.makedirs(DERIVED_DIR, exist_ok=True)

VARIANT_ORDER  = ["js_pure", "js_worker", "js_wasm", "js_worker_wasm", "js_wasm_simd", "js_worker_wasm_simd"]
WORKLOAD_ORDER = ["small_512x512_k5_p1", "medium_1920x1080_k15_p1", "large_1920x1080_k15_p10", "heavy_2560x1440_k31_p5"]
WORKLOAD_SHORT = {
    "small_512x512_k5_p1":     "Small",
    "medium_1920x1080_k15_p1": "Medium",
    "large_1920x1080_k15_p10": "Large",
    "heavy_2560x1440_k31_p5":  "Heavy",
}

df = pd.read_csv(SYSTEM_CSV)
df["variant"]  = pd.Categorical(df["variant"],  categories=VARIANT_ORDER, ordered=True)
df["workload"] = pd.Categorical(df["workload"],  categories=WORKLOAD_ORDER, ordered=True)
df["workload_short"] = df["workload"].map(WORKLOAD_SHORT)

rps = df[["variant", "workload", "workload_short", "poolSize", "concurrency",
          "throughput_rps", "is_worker"]].copy()
rps["throughput_rps"] = rps["throughput_rps"].round(3)

rps = rps.sort_values(["workload", "variant", "poolSize", "concurrency"])
rps.to_csv(os.path.join(DERIVED_DIR, "sys_01_rps_overview.csv"), index=False)

lb = df[["variant", "workload", "workload_short", "poolSize", "concurrency",
         "server_side_compute_mean", "server_side_queuewait_mean",
         "network_overhead_ms", "server_latency_mean", "client_latency_mean",
         "is_worker"]].copy()

lb["rest_ms"] = (lb["server_latency_mean"]
                 - lb["server_side_compute_mean"].fillna(0)
                 - lb["server_side_queuewait_mean"].fillna(0)).round(3)
lb["server_side_compute_mean"]   = lb["server_side_compute_mean"].round(3)
lb["server_side_queuewait_mean"] = lb["server_side_queuewait_mean"].round(3)
lb["network_overhead_ms"]        = lb["network_overhead_ms"].round(3)
lb["server_latency_mean"]        = lb["server_latency_mean"].round(3)
lb["client_latency_mean"]        = lb["client_latency_mean"].round(3)

lb = lb.sort_values(["workload", "variant", "poolSize", "concurrency"])
lb.to_csv(os.path.join(DERIVED_DIR, "sys_02_latency_breakdown.csv"), index=False)

rows = []
for wl in WORKLOAD_ORDER:
    sub = df[df["workload"] == wl]

    base_c1 = sub[(sub["variant"] == "js_pure") & (sub["concurrency"] == 1)]["throughput_rps"].mean()

    for _, row in sub.iterrows():
        c = row["concurrency"]
        v = row["variant"]
        rps_val = row["throughput_rps"]

        if c == 1:
            speedup = rps_val / base_c1 if base_c1 > 0 else np.nan
        else:
            base_cN = sub[(sub["variant"] == "js_worker") &
                          (sub["poolSize"] == c / 2) &
                          (sub["concurrency"] == c)]["throughput_rps"].mean()
            if v == "js_pure":
                speedup = np.nan
            else:
                speedup = rps_val / base_cN if (base_cN > 0 and not np.isnan(base_cN)) else np.nan

        rows.append({
            "variant": v,
            "workload": wl,
            "workload_short": WORKLOAD_SHORT[wl],
            "poolSize": row["poolSize"],
            "concurrency": c,
            "throughput_rps": round(rps_val, 3),
            "speedup": round(speedup, 3) if not np.isnan(speedup) else np.nan,
            "is_worker": row["is_worker"],
        })

speedup_df = pd.DataFrame(rows)
speedup_df["variant"]  = pd.Categorical(speedup_df["variant"],  categories=VARIANT_ORDER, ordered=True)
speedup_df["workload"] = pd.Categorical(speedup_df["workload"], categories=WORKLOAD_ORDER, ordered=True)
speedup_df = speedup_df.sort_values(["workload", "variant", "poolSize", "concurrency"])
speedup_df.to_csv(os.path.join(DERIVED_DIR, "sys_03_rps_speedup.csv"), index=False)

er = df[["variant", "workload", "workload_short", "poolSize", "concurrency",
         "client_err", "client_ok", "client_error_rate_percent",
         "server_requests_error_rate_percent", "is_worker"]].copy()

er["client_error_rate_percent"]          = er["client_error_rate_percent"].round(4)
er["server_requests_error_rate_percent"] = er["server_requests_error_rate_percent"].round(4)

er = er.sort_values(["workload", "variant", "poolSize", "concurrency"])
er.to_csv(os.path.join(DERIVED_DIR, "sys_04_error_rates.csv"), index=False)

el = df[["variant", "workload", "workload_short", "poolSize", "concurrency",
         "server_eventloop_lag_p95_after", "server_eventloop_lag_max_after",
         "server_eventloop_lag_p95_delta", "is_worker"]].copy()

el["server_eventloop_lag_p95_after"]  = el["server_eventloop_lag_p95_after"].round(2)
el["server_eventloop_lag_max_after"]  = el["server_eventloop_lag_max_after"].round(2)
el["server_eventloop_lag_p95_delta"]  = el["server_eventloop_lag_p95_delta"].round(2)

el = el.sort_values(["workload", "variant", "poolSize", "concurrency"])
el.to_csv(os.path.join(DERIVED_DIR, "sys_05_el_lag_load.csv"), index=False)

rows6 = []
for _, row in df.iterrows():
    v = row["variant"]
    c = row["concurrency"]
    ps = row["poolSize"]

    if row["is_worker"]:
        if ps != c:
            continue

    rows6.append({
        "variant": v,
        "workload": row["workload"],
        "workload_short": WORKLOAD_SHORT.get(row["workload"], row["workload"]),
        "concurrency": c,
        "poolSize": ps,
        "throughput_rps": round(row["throughput_rps"], 3),
        "client_latency_mean": round(row["client_latency_mean"], 3),
        "client_latency_p95":  round(row["client_latency_p95"],  3),
        "server_cpu_percent_after": round(row["server_cpu_percent_after"], 2),
        "is_worker": row["is_worker"],
    })

scale_df = pd.DataFrame(rows6)
scale_df["variant"]  = pd.Categorical(scale_df["variant"],  categories=VARIANT_ORDER, ordered=True)
scale_df["workload"] = pd.Categorical(scale_df["workload"], categories=WORKLOAD_ORDER, ordered=True)
scale_df = scale_df.sort_values(["workload", "variant", "concurrency"])
scale_df.to_csv(os.path.join(DERIVED_DIR, "sys_06_concurrency_scaling.csv"), index=False)
