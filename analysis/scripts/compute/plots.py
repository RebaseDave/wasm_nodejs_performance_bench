import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DERIVED_DIR = "../../results/tables/compute/"
OUTPUT_DIR  = "../../results/plots/compute/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
 
COLOR_MAP = {
    "js_pure":             "#92c5de",
    "js_wasm":             "#4393c3",
    "js_wasm_simd":        "#2166ac",
    "js_worker":           "#f4a582",
    "js_worker_wasm":      "#d6604d",
    "js_worker_wasm_simd": "#b2182b",
}
 
VARIANT_ORDER = [
    "js_pure", "js_worker",
    "js_wasm", "js_worker_wasm",
    "js_wasm_simd", "js_worker_wasm_simd",
]
VARIANT_LABELS = {
    "js_pure":             "JS Pure",
    "js_worker":           "JS Worker",
    "js_wasm":             "Wasm",
    "js_worker_wasm":      "Wasm Worker",
    "js_wasm_simd":        "Wasm SIMD",
    "js_worker_wasm_simd": "Wasm SIMD Worker",
}
 
WORKLOAD_ORDER = [
    "tiny_128x128_k3_p1",
    "small_512x512_k5_p1",
    "medium_1920x1080_k15_p1",
    "large_1920x1080_k15_p10",
    "heavy_2560x1440_k31_p5",
    "ultra_3840x2160_k31_p5",
]
WORKLOAD_LABELS = {
    "tiny_128x128_k3_p1":       "Tiny\n128×128\nk3p1",
    "small_512x512_k5_p1":      "Small\n512×512\nk5p1",
    "medium_1920x1080_k15_p1":  "Medium\n1920×1080\nk15p1",
    "large_1920x1080_k15_p10":  "Large\n1920×1080\nk15p10",
    "heavy_2560x1440_k31_p5":   "Heavy\n2560×1440\nk31p5",
    "ultra_3840x2160_k31_p5":   "Ultra\n3840×2160\nk31p5",
}
WORKLOAD_SHORT = {
    "tiny_128x128_k3_p1":       "Tiny",
    "small_512x512_k5_p1":      "Small",
    "medium_1920x1080_k15_p1":  "Medium",
    "large_1920x1080_k15_p10":  "Large",
    "heavy_2560x1440_k31_p5":   "Heavy",
    "ultra_3840x2160_k31_p5":   "Ultra",
}
 
FS_TITLE  = 12
FS_LABEL  = 11
FS_TICK   = 9
FS_LEGEND = 9
 

df01 = pd.read_csv(os.path.join(DERIVED_DIR, "01_single_mode_overview.csv"))
df01["workload"] = pd.Categorical(df01["workload"], categories=WORKLOAD_ORDER, ordered=True)
df01["variant"]  = pd.Categorical(df01["variant"],  categories=VARIANT_ORDER,  ordered=True)
df01 = df01.sort_values(["workload", "variant"])
 
WORKLOAD_TITLES = {
    "tiny_128x128_k3_p1":       "Tiny (128×128, k3, p1)",
    "small_512x512_k5_p1":      "Small (512×512, k5, p1)",
    "medium_1920x1080_k15_p1":  "Medium (1920×1080, k15, p1)",
    "large_1920x1080_k15_p10":  "Large (1920×1080, k15, p10)",
    "heavy_2560x1440_k31_p5":   "Heavy (2560×1440, k31, p5)",
    "ultra_3840x2160_k31_p5":   "Ultra (3840×2160, k31, p5)",
}
 
x_pos = np.arange(len(VARIANT_ORDER))
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
 
for ax, wl in zip(axes.flatten(), WORKLOAD_ORDER):
    sub  = df01[df01["workload"] == wl].set_index("variant").reindex(VARIANT_ORDER)
    vals = sub["throughput_mpix_sec"].values
    errs = sub["throughput_std"].values
    ax.bar(x_pos, vals, width=0.65,
           color=[COLOR_MAP[v] for v in VARIANT_ORDER],
           edgecolor="white", linewidth=0.5,
           yerr=errs, capsize=3,
           error_kw={"elinewidth": 1.0, "ecolor": "#555555"})
    ax.set_title(WORKLOAD_TITLES[wl], fontsize=FS_TICK + 1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER],
                       fontsize=FS_TICK - 1, rotation=30, ha="right")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=FS_TICK - 1)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
 
for ax in axes[:, 0]:
    ax.set_ylabel("Durchsatz [MPix/s]", fontsize=FS_TICK)
 
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[v], label=VARIANT_LABELS[v])
    for v in VARIANT_ORDER
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6,
           fontsize=FS_LEGEND, frameon=True, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Durchsatz nach Variante: Single-Thread", fontsize=FS_TITLE)
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_01_throughput_scaling.png"), dpi=150, bbox_inches="tight")
plt.close()
 
 
df03 = pd.read_csv(os.path.join(DERIVED_DIR, "03_wasm_vs_js_speedup.csv"))
df03["workload"] = pd.Categorical(df03["workload"], categories=WORKLOAD_ORDER, ordered=True)
 
hm_variants = [v for v in VARIANT_ORDER if v != "js_pure"]
pivot = (
    df03[df03["variant"].isin(hm_variants)]
    .pivot(index="variant", columns="workload", values="speedup")
    .reindex(index=hm_variants, columns=WORKLOAD_ORDER)
)
 
fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(pivot, ax=ax,
            annot=True, fmt=".2f",
            cmap="RdYlGn", center=1.0, vmin=0.7, vmax=1.4,
            linewidths=0.4,
            cbar_kws={"label": "Speedup vs. js_pure", "shrink": 0.8},
            xticklabels=[WORKLOAD_SHORT[w] for w in WORKLOAD_ORDER],
            yticklabels=[VARIANT_LABELS[v] for v in hm_variants],
            annot_kws={"size": FS_TICK})
ax.set_title("Speedup gegenüber js_pure: Single-Thread", fontsize=FS_TITLE)
ax.set_xlabel("Workload", fontsize=FS_LABEL)
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=FS_TICK, rotation=0)
ax.tick_params(axis="y", labelsize=FS_TICK, rotation=0)
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_02_speedup_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
 
 
df07 = pd.read_csv(os.path.join(DERIVED_DIR, "07_event_loop_and_cpu.csv"))
df07_w = df07[
    (df07["mode"] == "single") &
    (df07["variant"].isin(["js_worker", "js_worker_wasm", "js_worker_wasm_simd"]))
].copy()
df07_w["workload"] = pd.Categorical(df07_w["workload"], categories=WORKLOAD_ORDER, ordered=True)
df07_w = df07_w.sort_values(["workload", "variant"])
 
WORKER_VARIANTS_EL = ["js_worker", "js_worker_wasm", "js_worker_wasm_simd"]
x    = np.arange(len(WORKLOAD_ORDER))
offs = np.array([-1, 0, 1]) * 0.25
 
fig, ax = plt.subplots(figsize=(11, 5))
for i, variant in enumerate(WORKER_VARIANTS_EL):
    vals = (df07_w[df07_w["variant"] == variant]
            .set_index("workload").reindex(WORKLOAD_ORDER)["el_lag_max_ms"].fillna(0))
    ax.bar(x + offs[i], vals, width=0.25,
           color=COLOR_MAP[variant], label=VARIANT_LABELS[variant],
           edgecolor="white", linewidth=0.4)
 
ax.set_xticks(x)
ax.set_xticklabels([WORKLOAD_LABELS[w].replace("\n", " ") for w in WORKLOAD_ORDER],
                   fontsize=FS_TICK, rotation=15, ha="right")
ax.set_xlabel("Workload", fontsize=FS_LABEL)
ax.set_ylabel("Event-Loop-Lag Max [ms]", fontsize=FS_LABEL)
ax.set_title("Event-Loop-Lag (Maximum) – Worker-Varianten, Single Mode", fontsize=FS_TITLE)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, bbox_to_anchor=(1.01, 1), loc="upper left",
          frameon=True, title="Variante", title_fontsize=FS_TICK)
ax.grid(axis="y", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_03_worker_el_lag.png"), dpi=150, bbox_inches="tight")
plt.close()
 
 
df10 = pd.read_csv(os.path.join(DERIVED_DIR, "10_load_mode_overview.csv"))
 
PLOT_WORKLOAD_TITLES = {
    "tiny_128x128_k3_p1":       "Tiny\n128×128, k3, p1",
    "small_512x512_k5_p1":      "Small\n512×512, k5, p1",
    "medium_1920x1080_k15_p1":  "Medium\n1920×1080, k15, p1",
    "large_1920x1080_k15_p10":  "Large\n1920×1080, k15, p10",
    "heavy_2560x1440_k31_p5":   "Heavy\n2560×1440, k31, p5",
    "ultra_3840x2160_k31_p5":   "Ultra\n3840×2160, k31, p5",
}
 
LINE_CFG = {
    "js_worker":           ("--", COLOR_MAP["js_worker"]),
    "js_worker_wasm":      ("-",  COLOR_MAP["js_worker_wasm"]),
    "js_worker_wasm_simd": ("-",  COLOR_MAP["js_worker_wasm_simd"]),
}
LOAD_PLOT_VARIANTS = ["js_worker", "js_worker_wasm", "js_worker_wasm_simd"]
 
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
 
for ax, wl in zip(axes.flatten(), WORKLOAD_ORDER):
    sub = df10[df10["workload"] == wl].copy()
    for variant in LOAD_PLOT_VARIANTS:
        vsub = sub[sub["variant"] == variant].sort_values("concurrency")
        if vsub.empty:
            continue
        ls, col = LINE_CFG[variant]
        lbl = VARIANT_LABELS[variant] + (" (Referenz)" if variant == "js_worker" else "")
        ax.plot(vsub["concurrency"], vsub["throughput_mpix_sec"],
                marker="o", markersize=5, linewidth=1.8,
                color=col, linestyle=ls, label=lbl)
    ax.set_title(PLOT_WORKLOAD_TITLES[wl], fontsize=FS_TICK + 1)
    ax.set_xlabel("Worker-Pool-Größe", fontsize=FS_TICK)
    ax.set_xticks([1, 4, 8, 16])
    ax.set_xlim(0, 17)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=FS_TICK - 1)
    ax.grid(True, alpha=0.4)
 
for ax in axes[:, 0]:
    ax.set_ylabel("Durchsatz [MPix/s]", fontsize=FS_TICK)
 
handles, labels = axes.flatten()[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           fontsize=FS_LEGEND, frameon=True, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Durchsatz nach Worker-Pool-Größe: Multithreading",
             fontsize=FS_TITLE)
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_04_load_scaling.png"), dpi=150, bbox_inches="tight")
plt.close()
 
 
df10_wasm = df10[df10["variant"].isin(["js_worker_wasm", "js_worker_wasm_simd"])].copy()
 
POOL_ORDER    = [1, 4, 8, 16]
WASM_VARIANTS = ["js_worker_wasm", "js_worker_wasm_simd"]
 
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
 
for ax, variant in zip(axes, WASM_VARIANTS):
    sub   = df10_wasm[df10_wasm["variant"] == variant].copy()
    pivot = sub.pivot(index="workload", columns="concurrency", values="speedup_vs_js_worker")
    pivot = pivot.reindex(index=WORKLOAD_ORDER, columns=POOL_ORDER)
    pivot.index   = [WORKLOAD_SHORT[w] for w in WORKLOAD_ORDER]
    pivot.columns = [f"Pool {c}" for c in POOL_ORDER]
    sns.heatmap(pivot, ax=ax,
                annot=True, fmt=".2f",
                cmap="RdYlGn", center=1.0, vmin=0.7, vmax=1.4,
                linewidths=0.5,
                cbar_kws={"label": "Speedup vs. JS Worker", "shrink": 0.8},
                annot_kws={"size": FS_TICK})
    ax.set_title(VARIANT_LABELS[variant], fontsize=FS_TITLE)
    ax.set_xlabel("Worker-Pool-Größe", fontsize=FS_LABEL)
    ax.set_ylabel("Workload" if ax == axes[0] else "", fontsize=FS_LABEL)
    ax.tick_params(axis="x", labelsize=FS_TICK, rotation=0)
    ax.tick_params(axis="y", labelsize=FS_TICK, rotation=0)
 
fig.suptitle("Speedup vs. JS Worker: Multithreading",
             fontsize=FS_TITLE)
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_05_load_speedup_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
 
 
df04 = pd.read_csv(os.path.join(DERIVED_DIR, "04_worker_overhead.csv"))
df04["workload"] = pd.Categorical(df04["workload"], categories=WORKLOAD_ORDER, ordered=True)
df04["variant"]  = pd.Categorical(df04["variant"],  categories=VARIANT_ORDER,  ordered=True)
df04 = df04.sort_values(["workload", "variant"])
 
WORKER_VARIANTS = ["js_worker", "js_worker_wasm", "js_worker_wasm_simd"]
x       = np.arange(len(WORKLOAD_ORDER))
offsets = np.array([-1, 0, 1]) * 0.25
 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
for ax, metric, ylabel, title in [
    (axes[0], "overhead_percent", "Overhead-Anteil [%]",
     "Relativer Worker-Overhead\n(Anteil an Gesamtlatenz)"),
    (axes[1], "worker_overhead_ms", "Overhead [ms]",
     "Absoluter Worker-Overhead\n(IPC + Transfer)"),
]:
    for i, variant in enumerate(WORKER_VARIANTS):
        vals = (df04[df04["variant"] == variant]
                .set_index("workload").reindex(WORKLOAD_ORDER)[metric].values)
        ax.bar(x + offsets[i], vals, width=0.25,
               color=COLOR_MAP[variant], label=VARIANT_LABELS[variant],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([WORKLOAD_SHORT[w] for w in WORKLOAD_ORDER], fontsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
 
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           fontsize=FS_LEGEND, frameon=True, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Worker-Overhead: Single-Thread", fontsize=FS_TITLE)
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_06_worker_overhead.png"), dpi=150, bbox_inches="tight")
plt.close()
 
 
df01 = pd.read_csv(os.path.join(DERIVED_DIR, "01_single_mode_overview.csv"))
ref  = df01[df01["variant"] == "js_pure"].set_index("workload")["compute_mean_ms"]
 
RATIO_VARIANTS = ["js_wasm", "js_wasm_simd", "js_worker_wasm", "js_worker_wasm_simd"]
x       = np.arange(len(WORKLOAD_ORDER))
bw      = 0.2
offsets = np.linspace(-(len(RATIO_VARIANTS) - 1) / 2,
                       (len(RATIO_VARIANTS) - 1) / 2,
                       len(RATIO_VARIANTS)) * bw
 
fig, ax = plt.subplots(figsize=(11, 5))
 
for i, variant in enumerate(RATIO_VARIANTS):
    ratio = (df01[df01["variant"] == variant]
             .set_index("workload").reindex(WORKLOAD_ORDER)["compute_mean_ms"]
             / ref.reindex(WORKLOAD_ORDER)).values
    ax.bar(x + offsets[i], ratio, width=bw,
           color=COLOR_MAP[variant], label=VARIANT_LABELS[variant],
           edgecolor="white", linewidth=0.4)
 
ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="js_pure")
ax.set_xticks(x)
ax.set_xticklabels([WORKLOAD_SHORT[w] for w in WORKLOAD_ORDER], fontsize=FS_TICK)
ax.set_xlabel("Workload", fontsize=FS_LABEL)
ax.set_ylabel("Rechenzeit relativ zu js_pure", fontsize=FS_LABEL)
ax.set_title("Relative Rechenzeit Wasm-Varianten vs. js_pure: Single-Thread", fontsize=FS_TITLE)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.set_ylim(bottom=0)
ax.legend(fontsize=FS_LEGEND, bbox_to_anchor=(1.01, 1), loc="upper left",
          frameon=True, title="Variante", title_fontsize=FS_TICK)
ax.grid(axis="y", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_07_wasm_compute_ratio.png"), dpi=150, bbox_inches="tight")
plt.close()

df05 = pd.read_csv(os.path.join(DERIVED_DIR, "05_load_mode_concurrency.csv"))

WORKER_VARIANTS_P8 = ["js_worker", "js_worker_wasm", "js_worker_wasm_simd"]
df08 = df05[
    (df05["variant"].isin(WORKER_VARIANTS_P8)) &
    (df05["poolSize"] == 8) &
    (df05["concurrency"] == 8)
].copy()
df08["workload"] = pd.Categorical(df08["workload"], categories=WORKLOAD_ORDER, ordered=True)
df08["variant"]  = pd.Categorical(df08["variant"],  categories=WORKER_VARIANTS_P8, ordered=True)
df08 = df08.sort_values(["workload", "variant"])

x_pos   = np.arange(len(WORKLOAD_ORDER))
bw      = 0.25
offsets = np.array([-1, 0, 1]) * bw

fig, ax = plt.subplots(figsize=(11, 5))
for i, variant in enumerate(WORKER_VARIANTS_P8):
    sub  = df08[df08["variant"] == variant].set_index("workload").reindex(WORKLOAD_ORDER)
    vals = sub["throughput_mpix_sec"].values
    errs = sub["throughput_std"].values
    ax.bar(x_pos + offsets[i], vals, width=bw,
           color=COLOR_MAP[variant], label=VARIANT_LABELS[variant],
           edgecolor="white", linewidth=0.5,
           yerr=errs, capsize=3,
           error_kw={"elinewidth": 1.0, "ecolor": "#555555"})

ax.set_xticks(x_pos)
ax.set_xticklabels([WORKLOAD_SHORT[w] for w in WORKLOAD_ORDER], fontsize=FS_TICK)
ax.set_xlabel("Workload", fontsize=FS_LABEL)
ax.set_ylabel("Durchsatz [MPix/s]", fontsize=FS_LABEL)
ax.set_title(
    "Durchsatz - Worker-Varianten: Multithreading (PoolSize = 8)",
    fontsize=FS_TITLE
)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.set_ylim(bottom=0)
ax.legend(fontsize=FS_LEGEND, bbox_to_anchor=(1.01, 1), loc="upper left",
          frameon=True, title="Variante", title_fontsize=FS_TICK)
ax.grid(axis="y", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_08_worker_pool8_throughput.png"),
            dpi=150, bbox_inches="tight")
plt.close()

MAIN_THREAD_VARIANTS = ["js_pure", "js_wasm", "js_wasm_simd"]

df09 = pd.read_csv(os.path.join(DERIVED_DIR, "09_memory_usage.csv"))
df09 = df09[df09["variant"].isin(MAIN_THREAD_VARIANTS)].copy()
df09["workload"] = pd.Categorical(df09["workload"], categories=WORKLOAD_ORDER, ordered=True)
df09["variant"]  = pd.Categorical(df09["variant"],  categories=MAIN_THREAD_VARIANTS, ordered=True)
df09 = df09.sort_values(["workload", "variant"])

n_variants = len(MAIN_THREAD_VARIANTS)
bw = 0.25
x = np.arange(len(WORKLOAD_ORDER))
offsets = np.linspace(-(n_variants - 1) / 2, (n_variants - 1) / 2, n_variants) * bw

fig, ax = plt.subplots(figsize=(11, 5))

for i, variant in enumerate(MAIN_THREAD_VARIANTS):
    sub  = df09[df09["variant"] == variant].set_index("workload").reindex(WORKLOAD_ORDER)
    vals = sub["mem_peak_rss_mb"].values
    errs = sub["mem_peak_rss_std"].values
    ax.bar(x + offsets[i], vals, width=bw,
           color=COLOR_MAP[variant], label=VARIANT_LABELS[variant],
           edgecolor="white", linewidth=0.4,
           yerr=errs, capsize=3,
           error_kw={"elinewidth": 0.8, "ecolor": "#555555"})

ax.set_xticks(x)
ax.set_xticklabels([WORKLOAD_SHORT[w] for w in WORKLOAD_ORDER], fontsize=FS_TICK)
ax.set_xlabel("Workload", fontsize=FS_LABEL)
ax.set_ylabel("Peak RSS (MB)", fontsize=FS_LABEL)
ax.set_title("Speicherverbrauch (Peak RSS): Main-Thread-Varianten", fontsize=FS_TITLE)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.4)
ax.set_axisbelow(True)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[v], label=VARIANT_LABELS[v])
    for v in MAIN_THREAD_VARIANTS
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=FS_LEGEND, frameon=True, bbox_to_anchor=(0.5, -0.04))

fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_09_memory_rss.png"), dpi=150, bbox_inches="tight")
plt.close()