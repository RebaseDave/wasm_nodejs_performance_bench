const fs = require("fs");
const path = require("path");
const os = require("os");
const { performance, monitorEventLoopDelay } = require("perf_hooks");

const { buildWorkloads } = require("./workloads");

const jsPure = require("../variants/js_pure");
const jsWasm = require("../variants/js_wasm");
const jsWasmSimd = require("../variants/js_wasm_simd");
const jsWorker = require("../variants/js_worker");
const jsWorkerWasm = require("../variants/js_worker_wasm");
const jsWorkerWasmSimd = require("../variants/js_worker_wasm_simd");

const VARIANTS = [jsPure, jsWasm, jsWasmSimd, jsWorker, jsWorkerWasm, jsWorkerWasmSimd];

function nowIsoCompact() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return (d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "_" + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds()));
}
//for cooldown
function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}

function mean(arr) {
    let s = 0;
    for (const x of arr) s += x;
    return s / arr.length;
}

function percentile(sortedArr, p) {
    const idx = (sortedArr.length - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) {
        return sortedArr[lo];
    }
    const w = idx - lo;
    return sortedArr[lo] * (1 - w) + sortedArr[hi] * w;
}

function statsCompact(samples) {
    const sorted = [...samples].sort((a, b) => a - b);
    return {
        mean: mean(samples),
        p95: percentile(sorted, 0.95)
    };
}
//normalizes variant output
function unwrapVariantResult(res) {
    if (res instanceof Uint8Array) {
        return { out: res, timings: null };
    }
    if (res && typeof res === "object" && res.out instanceof Uint8Array) {
        return { out: res.out, timings: res.timings || null };
    }
    return { out: res, timings: null };
}
//runs one workload for one variant and saves result metrics
async function runWorkloadForVariant(variant, workload, cfg) {
    const { width, height, kernelSize, inputs, name } = workload;
    //check variant and mode
    const isWorkerVariant = variant.isWorker === true;
    const mode = cfg.mode ?? "single";
    //set poolsize for worker-variants
    const poolSize = isWorkerVariant && Number.isInteger(cfg.variantOpts?.size)
        ? cfg.variantOpts.size : 1;

    const concurrency = (mode === "load" && isWorkerVariant)
        ? Math.max(1, Math.min(poolSize, inputs.length)) : 1;

    const passes = Number.isInteger(workload.passes) ? workload.passes : 1;
    //metric variables
    let peak = { rss: 0, heapUsed: 0 };
    let sampler = null;
    let memoryBefore = null;
    let cpuBeforeBlock = null;
    let cpuBlock = null;
    let t0Block = null;
    let t1Block = null;
    let elLag = null;
    let latenciesMs = null;
    let computeMsArr = null;
    let roundtripMsArr = null;
    //elu-monitor
    const eld = monitorEventLoopDelay({ resolution: 10 });
    //rss sampling
    try {
        if (cfg.samplePeakMemory) {
            const snap = process.memoryUsage();
            peak.rss = snap.rss;
            peak.heapUsed = snap.heapUsed;

            sampler = setInterval(() => {
                const m = process.memoryUsage();
                if (m.rss > peak.rss) {
                    peak.rss = m.rss;
                }
                if (m.heapUsed > peak.heapUsed) {
                    peak.heapUsed = m.heapUsed;
                }
            }, cfg.peakSampleMs);
        }
        //base values before run
        memoryBefore = process.memoryUsage();
        cpuBeforeBlock = process.cpuUsage();
        eld.enable();
        t0Block = performance.now();

        latenciesMs = new Array(inputs.length);
        computeMsArr = new Array(inputs.length).fill(null);
        roundtripMsArr = new Array(inputs.length);
        //runs variant with workload
        async function runOne(input, sampleIndex) {
            const t0Roundtrip = performance.now();

            const variantOpts = Object.assign({}, cfg.variantOpts || {}, {
                returnTimings: true,
                passes: passes
            });

            const res = await variant.run(input, width, height, kernelSize, variantOpts);
            const unwrapped = unwrapVariantResult(res);

            const t1Roundtrip = performance.now();
            const roundtripMs = t1Roundtrip - t0Roundtrip;

            roundtripMsArr[sampleIndex] = roundtripMs;
            //extract isolated compute-time
            const t = unwrapped.timings;
            if (t && typeof t === "object" && Number.isFinite(t.computeMs)) {
                computeMsArr[sampleIndex] = t.computeMs;
            }

            return roundtripMs;
        }
        //slot-based parallelization
        let nextIndex = 0;

        async function slotLoop() {
            while (true) {
                const i = nextIndex++;
                if (i >= inputs.length) {
                    break;
                }
                latenciesMs[i] = await runOne(inputs[i], i);
            }
        }

        const slots = [];
        for (let i = 0; i < concurrency; i++) {
            slots.push(slotLoop());
        }
        await Promise.all(slots);

        t1Block = performance.now();
        eld.disable();
        //read elu values
        try {
            const nsToMs = (ns) => (Number.isFinite(ns) ? ns / 1e6 : 0);
            elLag = {
                p95: nsToMs(eld.percentile(95)),
                p99: nsToMs(eld.percentile(99)),
                max: nsToMs(eld.max)
            };
        } catch (e) {
            elLag = { p95: 0, p99: 0, max: 0 };
        }

        cpuBlock = process.cpuUsage(cpuBeforeBlock);

    } catch (error) {
        console.error(`Error in workload ${workload.name}:`, error);
        throw error;
    } finally {
        try {
            eld.disable();
        } catch (e) {
        }
        if (sampler) {
            clearInterval(sampler);
        }
        //save peaks for memory
        if (cfg.samplePeakMemory) {
            const snap = process.memoryUsage();
            peak.rss = Math.max(peak.rss, snap.rss);
            peak.heapUsed = Math.max(peak.heapUsed, snap.heapUsed);
        }
    }
    //derive metrics
    const wallMs = t1Block - t0Block;
    const cpuMicros = cpuBlock.user + cpuBlock.system;
    const pixelsPerIter = width * height;
    const totalPixels = pixelsPerIter * inputs.length;
    const effectivePixels = totalPixels * passes;
    const mpixPerSec = (effectivePixels / 1e6) / (wallMs / 1000);

    return {
        workload: name,
        width,
        height,
        kernelSize,
        iterations: inputs.length,
        passes,
        mode,
        concurrency,
        poolSize,
        isWorkerVariant,
        latencyMs: statsCompact(latenciesMs),
        latencyBreakdown: {
            roundtrip: statsCompact(roundtripMsArr),
            compute: (computeMsArr && computeMsArr.some(v => v != null))
                ? statsCompact(computeMsArr.filter(v => v != null)) : null
        },
        eventLoopLag: elLag,
        wallMs,
        cpu: {
            totalMicros: cpuMicros,
            cpuToWallPercent: (cpuMicros / (wallMs * 1000)) * 100
        },
        throughput: {
            mpixPerSec
        },
        memory: {
            peak: cfg.samplePeakMemory ? peak : null
        }
    };
}
//warmup-runs
async function warmupVariant(variant, cfg) {
    const warmWorkloads = buildWorkloads({ iterations: cfg.warmupIters, baseSeed: cfg.baseSeed });

    const isWorkerVariant = variant.isWorker === true;
    const mode = cfg.mode ?? "single";
    const poolSize = isWorkerVariant && Number.isInteger(cfg.variantOpts?.size)
        ? cfg.variantOpts.size : 1;
    const variantOptsBase = cfg.variantOpts || {};

    for (const w of warmWorkloads) {
        const variantOpts = Object.assign({}, variantOptsBase, {
            returnTimings: false,
            passes: Number.isInteger(w.passes) ? w.passes : 1
        });

        const concurrency = (mode === "load" && isWorkerVariant)
            ? Math.max(1, Math.min(poolSize, w.inputs.length)) : 1;

        let nextIndex = 0;

        async function slotLoop() {
            while (true) {
                const i = nextIndex++;
                if (i >= w.inputs.length) {
                    break;
                }
                await variant.run(w.inputs[i], w.width, w.height, w.kernelSize, variantOpts);
            }
        }
        const slots = [];
        for (let i = 0; i < concurrency; i++) {
            slots.push(slotLoop());
        }
        await Promise.all(slots);
    }
}
//runs warmup and all workloads for one variant
async function runVariant(variant, cfg) {
    const workloads = buildWorkloads({ iterations: cfg.iterations, baseSeed: cfg.baseSeed });
    await warmupVariant(variant, cfg);

    try {
        const results = [];
        for (const wl of workloads) {
            results.push(await runWorkloadForVariant(variant, wl, cfg));
        }
        return results;
    } finally {
        if (typeof variant.close === "function") {
            await variant.close();
        }
    }
}
//generate output filenames
function outFileName(cfg, runTag) {
    const mode = cfg.mode ?? "single";
    const size = cfg.variantOpts?.size ?? 1;
    return `benchmark_${runTag}_${mode}_size${size}_${nowIsoCompact()}.json`
}
//full benchmark run
async function benchmarkOnce(cfg, runTag) {
    const mode = cfg.mode ?? "single";
    //system metadata
    const meta = {
        timestamp: new Date().toISOString(),
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        cpus: {
            count: os.cpus().length,
            model: os.cpus()[0]?.model || `unknown`,
        },
        memory: {
            totalMB: Math.round(os.totalmem() / (1024 ** 2)),
        },
        cfg,
        runTag
    };

    const all = { meta, variants: {} };
    const cooldownMs = Number.isInteger(cfg.cooldownMs) ? cfg.cooldownMs : 500;
    const activeVariants = VARIANTS.filter(v => {
        if (mode === "single") {
            return true;
        }
        return v.name === "js_pure" || v.isWorker === true;
    });

    for (let va = 0; va < activeVariants.length; va++) {
        const variant = activeVariants[va];
        console.log(`\n=== [${runTag}] Running variant: ${variant.name} (mode=${cfg.mode}, size=${cfg.variantOpts?.size}) ===`);
        all.variants[variant.name] = await runVariant(variant, cfg);
        //cooldown
        if (cooldownMs > 0 && va < activeVariants.length - 1) {
            console.log(`Cooldown: ${cooldownMs}ms`);
            await sleep(cooldownMs);
        }
    }
    //save results as JSON
    const outDir = path.join(__dirname, "results");
    fs.mkdirSync(outDir, { recursive: true });

    const outFile = path.join(outDir, outFileName(cfg, runTag));
    fs.writeFileSync(outFile, JSON.stringify(all, null, 2));
    console.log(`\nSaved results to: ${outFile}`);
}

//benchmark-configuration
function buildTestPlan() {
    const poolSizes = [1, 4, 8, 16];
    const singleModeReps = 1;
    const loadModeReps = 1;

    const warmupIters = 1;
    //iterations should be a multiple of poolSizes
    const iterations = 16;
    const cooldownMs = 100;

    //cfg - tiny
/*     const warmupIters = 50;
    const iterations = 500;
    const cooldownMs = 200; */

    const peakSampleMs = 200;
    const plan = [];
    //sequencial tasks
    for (let r = 1; r <= singleModeReps; r++) {
        plan.push({
            tag: `single_r${r}`,
            cfg: {
                mode: "single",
                iterations,
                warmupIters,
                baseSeed: 42,
                samplePeakMemory: true,
                peakSampleMs,
                cooldownMs,
                variantOpts: {
                    size: 4,
                    enforceTransfer: true
                }
            }
        });
    }
    //parallel tasks
    for (const size of poolSizes) {
        for (let r = 1; r <= loadModeReps; r++) {
            plan.push({
                tag: `load_size${size}_r${r}`,
                cfg: {
                    mode: "load",
                    iterations,
                    warmupIters,
                    baseSeed: 42,
                    samplePeakMemory: true,
                    peakSampleMs,
                    cooldownMs,
                    variantOpts: {
                        size,
                        enforceTransfer: true
                    }
                }
            });
        }
    }

    return plan;
}

async function main() {
    const plan = buildTestPlan();

    console.log(`\nPlanned runs: ${plan.length}`);
    for (let i = 0; i < plan.length; i++) {
        const { tag, cfg } = plan[i];
        console.log(`\n>>> RUN ${i + 1}/${plan.length}: ${tag}`);
        await benchmarkOnce(cfg, tag);
    }
}

main().catch((e) => {
    console.error(e);
    process.exitCode = 1;
});