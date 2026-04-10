import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

SYSTEM_CSV  = "../../processed_data/system_aggregated.csv"
DERIVED_DIR = "../../results/tables/system/"
OUTPUT_DIR  = "../../results/plots/system/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
FS_TITLE  = 12
FS_LABEL  = 11
FS_TICK   = 9
FS_LEGEND = 9

COLOR_MAP = {
    "js_pure":             "#92c5de",
    "js_wasm":             "#4393c3",
    "js_wasm_simd":        "#2166ac",
    "js_worker":           "#f4a582",
    "js_worker_wasm":      "#d6604d",
    "js_worker_wasm_simd": "#b2182b",
}
LABEL_MAP = {
    "js_pure":             "JS Pure",
    "js_wasm":             "Wasm",
    "js_wasm_simd":        "Wasm SIMD",
    "js_worker":           "JS Worker",
    "js_worker_wasm":      "Wasm Worker",
    "js_worker_wasm_simd": "Wasm SIMD Worker",
}
VARIANT_ORDER  = ["js_pure", "js_worker", "js_wasm", "js_worker_wasm", "js_wasm_simd", "js_worker_wasm_simd"]
WORKLOAD_ORDER = ["small_512x512_k5_p1", "medium_1920x1080_k15_p1", "large_1920x1080_k15_p10", "heavy_2560x1440_k31_p5"]
WORKLOAD_SHORT = {
    "small_512x512_k5_p1":     "Small\n512×512, k5, p1",
    "medium_1920x1080_k15_p1": "Medium\n1920×1080, k15, p1",
    "large_1920x1080_k15_p10": "Large\n1920×1080, k15, p10",
    "heavy_2560x1440_k31_p5":  "Heavy\n2560×1440, k31, p5",
}
WORKLOAD_SHORT_TITLE = {
    "small_512x512_k5_p1":     "Small",
    "medium_1920x1080_k15_p1": "Medium",
    "large_1920x1080_k15_p10": "Large",
    "heavy_2560x1440_k31_p5":  "Heavy",
}

df_raw = pd.read_csv(SYSTEM_CSV)
df_raw["variant"]  = pd.Categorical(df_raw["variant"],  categories=VARIANT_ORDER, ordered=True)
df_raw["workload"] = pd.Categorical(df_raw["workload"], categories=WORKLOAD_ORDER, ordered=True)

SNAPSHOT_POOL = 8

c1_all = df_raw[
    ((df_raw["is_worker"] == False) & (df_raw["concurrency"] == 1) & (df_raw["poolSize"] == -1)) |
    ((df_raw["is_worker"] == True)  & (df_raw["concurrency"] == 1) & (df_raw["poolSize"] == 1))
].copy()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Durchsatz nach Variante – Single-Thread \n(c=1, poolSize=1)",
             fontsize=FS_TITLE, y=1.01)

for ax, wl in zip(axes.flat, WORKLOAD_ORDER):
    sub = c1_all[c1_all["workload"] == wl].sort_values("variant")
    colors = [COLOR_MAP[v] for v in sub["variant"]]
    labels = [LABEL_MAP[v] for v in sub["variant"]]
    bars = ax.bar(range(len(sub)), sub["throughput_rps"], color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, fontsize=FS_TICK - 1, rotation=30, ha="right")
    ax.set_title(WORKLOAD_SHORT_TITLE[wl], fontsize=FS_LABEL)
    ax.set_ylabel("Durchsatz (RPS)", fontsize=FS_LABEL)
    ax.set_ylim(0, sub["throughput_rps"].max() * 1.2)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_sys_01a_rps_mainthread.png"), dpi=150, bbox_inches="tight")
plt.close()

WORKER_VARIANTS_LIST = ["js_worker", "js_worker_wasm", "js_worker_wasm_simd"]

rps_w = df_raw[
    (df_raw["is_worker"] == True) &
    (df_raw["concurrency"] == SNAPSHOT_POOL) &
    (df_raw["poolSize"] == SNAPSHOT_POOL)
].copy()

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(f"Durchsatz - Worker-Varianten \npoolSize={SNAPSHOT_POOL}, c={SNAPSHOT_POOL}",
             fontsize=FS_TITLE, y=1.03)

for ax, wl in zip(axes, WORKLOAD_ORDER):
    sub = rps_w[(rps_w["workload"] == wl) &
                (rps_w["variant"].isin(WORKER_VARIANTS_LIST))].sort_values("variant")
    colors = [COLOR_MAP[v] for v in sub["variant"]]
    labels = [LABEL_MAP[v] for v in sub["variant"]]
    bars = ax.bar(range(len(sub)), sub["throughput_rps"], color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, fontsize=FS_TICK - 1, rotation=30, ha="right")
    ax.set_title(WORKLOAD_SHORT_TITLE[wl], fontsize=FS_LABEL)
    ax.set_ylabel("Durchsatz (RPS)", fontsize=FS_LABEL)
    ax.set_ylim(0, sub["throughput_rps"].max() * 1.2 if not sub.empty else 1)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, sub["throughput_rps"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_height() * 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_sys_01b_rps_worker_load.png"), dpi=150, bbox_inches="tight")
plt.close()

POOL_SIZES     = [1, 4, 8, 16]
CONC_MAP       = {1: 2, 4: 8, 8: 16, 16: 32}
WORKER_VARIANTS = ["js_worker", "js_worker_wasm", "js_worker_wasm_simd"]
 
df_scale = df_raw[
    (df_raw["is_worker"] == True) &
    (df_raw["variant"].isin(WORKER_VARIANTS))
].copy()

df_scale = df_scale[df_scale["concurrency"] == df_scale["poolSize"] * 2]
 
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Durchsatz nach Worker-Pool-Größe: Multithreading\n(concurrency = 2 × poolSize)",
             fontsize=FS_TITLE, y=1.01)
 
handles, plotted_labels = [], []
for ax, wl in zip(axes.flat, WORKLOAD_ORDER):
    for v in WORKER_VARIANTS:
        sub = df_scale[
            (df_scale["workload"] == wl) &
            (df_scale["variant"] == v)
        ].sort_values("poolSize")
        if sub.empty:
            continue
        ls = "--" if v == "js_worker" else "-"
        lw = 1.5 if v == "js_worker" else 2.0
        line, = ax.plot(sub["poolSize"], sub["throughput_rps"],
                        color=COLOR_MAP[v], linestyle=ls, linewidth=lw,
                        marker="o", markersize=5, label=LABEL_MAP[v])
        if LABEL_MAP[v] not in plotted_labels:
            handles.append(line)
            plotted_labels.append(LABEL_MAP[v])
 
    ax.set_title(WORKLOAD_SHORT_TITLE[wl], fontsize=FS_LABEL)
    ax.set_xlabel("Worker-Pool-Größe", fontsize=FS_LABEL)
    ax.set_ylabel("Durchsatz (RPS)", fontsize=FS_LABEL)
    ax.set_xticks(POOL_SIZES)
    ax.set_xticklabels([f"{p}\n(c={CONC_MAP[p]})" for p in POOL_SIZES], fontsize=FS_TICK)
    ax.set_ylim(0)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.4)
 
fig.legend(handles, plotted_labels, fontsize=FS_LEGEND, loc="lower center",
           ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_sys_02_rps_scaling.png"), dpi=150, bbox_inches="tight")
plt.close()

lb = pd.read_csv(os.path.join(DERIVED_DIR, "sys_02_latency_breakdown.csv"))
lb["variant"]  = pd.Categorical(lb["variant"],  categories=VARIANT_ORDER, ordered=True)
lb["workload"] = pd.Categorical(lb["workload"], categories=WORKLOAD_ORDER, ordered=True)

CONC_A, CONC_B = 8, 16

lb_a = lb[(lb["is_worker"] == True) & (lb["poolSize"] == SNAPSHOT_POOL) & (lb["concurrency"] == CONC_A)]
lb_b = lb[(lb["is_worker"] == True) & (lb["poolSize"] == SNAPSHOT_POOL) & (lb["concurrency"] == CONC_B)]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"Latenz-Aufschlüsselung: Worker-Varianten (poolSize={SNAPSHOT_POOL})",
             fontsize=FS_TITLE, y=1.01)

BAR_W = 0.38
GAP   = 0.05

for ax, wl in zip(axes.flat, WORKLOAD_ORDER):
    sub_a = lb_a[lb_a["workload"] == wl].sort_values("variant")
    sub_b = lb_b[lb_b["workload"] == wl].sort_values("variant")

    n = len(sub_a)
    x = np.arange(n)

    for offset, sub, alpha in [(-(BAR_W + GAP) / 2, sub_a, 1.0),
                                ( (BAR_W + GAP) / 2, sub_b, 0.55)]:
        comp       = sub["server_side_compute_mean"].fillna(0).values
        queue      = sub["server_side_queuewait_mean"].fillna(0).values
        rest       = np.clip(sub["rest_ms"].values, 0, None)
        colors_var = [COLOR_MAP[v] for v in sub["variant"]]

        for i, (c_val, q_val, r_val, col) in enumerate(zip(comp, queue, rest, colors_var)):
            ax.bar(x[i] + offset, c_val, width=BAR_W, color=col,
                   alpha=alpha, edgecolor="white", linewidth=0.4)
            ax.bar(x[i] + offset, q_val, width=BAR_W, color="#555555",
                   alpha=alpha, bottom=c_val,
                   edgecolor="white", linewidth=0.4)
            ax.bar(x[i] + offset, r_val, width=BAR_W, color="#f0a500",
                   alpha=alpha, bottom=c_val + q_val,
                   edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_MAP[v] for v in sub_a["variant"]],
                       fontsize=FS_TICK - 1, rotation=30, ha="right")
    ax.set_title(WORKLOAD_SHORT_TITLE[wl], fontsize=FS_LABEL)
    ax.set_ylabel("Latenz [ms]", fontsize=FS_LABEL)
    ax.set_ylim(0)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

seg_handles = [
    plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=1.0),
    plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=1.0),
    plt.Rectangle((0, 0), 1, 1, color="#f0a500", alpha=1.0),
    plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=1.0),
    plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=0.55),
]
fig.legend(seg_handles,
           ["Compute (Varianten-Farbe)", "Queue-Wait", "Rest (HTTP-Overhead u.A.)",
            f"c={CONC_A}", f"c={CONC_B} (transparent)"],
           fontsize=FS_LEGEND, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.04), frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_sys_03_latency_breakdown.png"), dpi=150, bbox_inches="tight")
plt.close()

spd = pd.read_csv(os.path.join(DERIVED_DIR, "sys_03_rps_speedup.csv"))
spd["variant"]  = pd.Categorical(spd["variant"],  categories=VARIANT_ORDER, ordered=True)
spd["workload"] = pd.Categorical(spd["workload"], categories=WORKLOAD_ORDER, ordered=True)
 
spd_filtered = spd[spd["concurrency"] == spd["poolSize"] * 2].copy()
 
wasm_variants = ["js_worker_wasm", "js_worker_wasm_simd"]
wasm_titles   = {"js_worker_wasm": "Wasm Worker", "js_worker_wasm_simd": "Wasm SIMD Worker"}
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Speedup vs. JS Worker: Multithreading (RPS)\n(concurrency = 2 × poolSize)",
             fontsize=FS_TITLE)
 
for ax, v in zip(axes, wasm_variants):
    sub = spd_filtered[spd_filtered["variant"] == v].copy()
    pivot = sub.pivot(index="workload", columns="poolSize", values="speedup")
    pivot = pivot.reindex(index=WORKLOAD_ORDER)
    pivot.index   = [WORKLOAD_SHORT_TITLE[w] for w in WORKLOAD_ORDER if w in pivot.index]
    pivot.columns = [f"Pool {int(c)}\n(c={CONC_MAP[int(c)]})" for c in pivot.columns]
 
    sns.heatmap(pivot, ax=ax,
                annot=True, fmt=".2f",
                cmap="RdYlGn", center=1.0, vmin=0.7, vmax=1.4,
                linewidths=0.4,
                cbar_kws={"label": "Speedup vs. JS Worker", "shrink": 0.8},
                annot_kws={"size": FS_TICK})
    ax.set_title(wasm_titles[v], fontsize=FS_TITLE)
    ax.set_xlabel("Worker-Pool-Größe", fontsize=FS_LABEL)
    ax.set_ylabel("Workload" if ax == axes[0] else "", fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK, rotation=0)
    ax.tick_params(axis="y", labelsize=FS_TICK, rotation=0)
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_sys_04_speedup_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

el = pd.read_csv(os.path.join(DERIVED_DIR, "sys_05_el_lag_load.csv"))
el["variant"]  = pd.Categorical(el["variant"],  categories=VARIANT_ORDER, ordered=True)
el["workload"] = pd.Categorical(el["workload"], categories=WORKLOAD_ORDER, ordered=True)

el_snap = el[
    (el["is_worker"] == True) &
    (el["poolSize"] == SNAPSHOT_POOL) &
    (el["concurrency"] == SNAPSHOT_POOL)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Event-Loop-Lag unter HTTP-Last: Worker-Varianten\n"
             f"(poolSize={SNAPSHOT_POOL}, c={SNAPSHOT_POOL} | Balken = p95, Fehlerbalken = Max)",
             fontsize=FS_TITLE, y=1.01)

for ax, wl in zip(axes.flat, WORKLOAD_ORDER):
    sub = el_snap[el_snap["workload"] == wl].sort_values("variant")
    labels = [LABEL_MAP[v] for v in sub["variant"]]
    colors = [COLOR_MAP[v] for v in sub["variant"]]
    x = np.arange(len(sub))

    p95 = sub["server_eventloop_lag_p95_after"].values
    mx  = sub["server_eventloop_lag_max_after"].values
    overflow = np.clip(mx - p95, 0, None)

    ax.bar(x, p95,      width=0.6, color=colors,    alpha=0.9,
           edgecolor="white", linewidth=0.5)
    ax.bar(x, overflow, width=0.6, color="#cccccc",  alpha=0.9, bottom=p95,
           edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK - 1, rotation=30, ha="right")
    ax.set_title(WORKLOAD_SHORT_TITLE[wl], fontsize=FS_LABEL)
    ax.set_ylabel("EL-Lag [ms]", fontsize=FS_LABEL)
    ax.set_ylim(0)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

handles = [
    plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=0.9),
    plt.Rectangle((0, 0), 1, 1, color="#cccccc", alpha=0.9),
]
fig.legend(handles, ["p95 EL-Lag", "Max EL-Lag"],
           fontsize=FS_LEGEND, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.04), frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_sys_05_el_lag.png"), dpi=150, bbox_inches="tight")
plt.close()