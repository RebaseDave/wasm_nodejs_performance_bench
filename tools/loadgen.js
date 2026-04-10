const http = require("http");
const https = require("https");
const { URL } = require("url");
const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");
const os = require("os");
const { TestImageGenerator } = require("./test-image-generator");

//persistent connection pools to avoid tcp overhead per request
const httpAgent = new http.Agent({
    keepAlive: true,
    keepAliveMsecs: 1000,
    maxSockets: 256,
    maxFreeSockets: 64
});

const httpsAgent = new https.Agent({
    keepAlive: true,
    keepAliveMsecs: 1000,
    maxSockets: 256,
    maxFreeSockets: 64
});

function getAgent(u) {
    return u.protocol === "https:" ? httpsAgent : httpAgent;
}

function parseArgs(argv) {
    const args = {};
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (!a.startsWith("--")) {
            continue;
        }
        const key = a.slice(2);
        const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : true;
        args[key] = val;
    }
    return args;
}

function toInt(v, name) {
    const n = Number(v);
    if (!Number.isInteger(n)) {
        throw new Error(`${name} must be of type: integer`);
    }
    return n;
}
//for cooldown
function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}

function mean(arr) {
    let s = 0;
    for (const x of arr) {
        s += x;
    }
    return arr.length ? s / arr.length : 0;
}

function quantile(sortedArr, q) {
    if (!sortedArr.length) {
        return 0;
    }
    const pos = (sortedArr.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;

    if (sortedArr[base + 1] === undefined) {
        return sortedArr[base];
    }
    return sortedArr[base] + rest * (sortedArr[base + 1] - sortedArr[base]);
}

function summarizeLatencies(lat) {
    if (!lat.length) {
        return { mean: 0, p50: 0, p95: 0, p99: 0, max: 0 }
    }
    const sorted = [...lat].sort((a, b) => a - b);
    return {
        mean: mean(lat),
        p50: quantile(sorted, 0.5),
        p95: quantile(sorted, 0.95),
        p99: quantile(sorted, 0.99),
        max: sorted[sorted.length - 1]
    };
}

function getHttpModule(u) {
    return u.protocol === "https:" ? https : http;
}
//GET request returning parsed JSON
function httpGetJson(serverBase, pathname) {
    const u = new URL(serverBase);
    u.pathname = pathname;
    u.search = "";

    const mod = getHttpModule(u);

    return new Promise((resolve, reject) => {
        const req = mod.request(u, { method: "GET", agent: getAgent(u), headers: { "Accept": "application/json" } },
            res => {
                const chunks = [];
                res.on("data", (c) => chunks.push(c));
                res.on("end", () => {
                    const body = Buffer.concat(chunks).toString("utf8");
                    if (res.statusCode !== 200) {
                        return reject(new Error(`GET ${pathname} -> ${res.statusCode}: ${body}`));
                    }
                    try {
                        resolve(JSON.parse(body));
                    } catch (e) {
                        reject(new Error(`Failed to parse JSON from ${pathname}: ${e.message}\nbody=${body}`));
                    }
                });
            }
        );
        req.on("error", reject);
        req.end();
    });
}
//POST request with optional body, returns parsed JSON response
function httpPostJson(serverBase, pathname, bodyObj = null) {
    const u = new URL(serverBase);
    u.pathname = pathname;
    u.search = "";

    const mod = getHttpModule(u);
    const payload = bodyObj ? Buffer.from(JSON.stringify(bodyObj), "utf8") : null;

    const options = {
        method: "POST",
        agent: getAgent(u),
        headers: {
            "Accept": "application/json",
            ...(payload
                ? { "Content-Type": "application/json", "Content-Length": payload.byteLength }
                : { "Content-Length": 0 })
        }
    };

    return new Promise((resolve, reject) => {
        const req = mod.request(u, options, (res) => {
            const chunks = [];
            res.on("data", (c) => chunks.push(c));
            res.on("end", () => {
                const text = Buffer.concat(chunks).toString("utf8");
                if (res.statusCode !== 200) {
                    return reject(new Error(`POST ${pathname} -> ${res.statusCode}: ${text}`));
                }
                try {
                    resolve(text ? JSON.parse(text) : {});
                } catch (e) {
                    reject(new Error(`Failed to parse JSON from POST ${pathname}: ${e.message}\nbody=${text}`));
                }
            });
        });
        req.on("error", reject);
        if (payload) req.write(payload);
        req.end();
    });
}
//POST raw RGBA binary to /blur/:variant, returning timingheaders and status
function httpPostBinary(serverBase, pathWithQuery, payloadBuf) {
    const u = new URL(serverBase);
    const mod = getHttpModule(u);

    const options = {
        protocol: u.protocol,
        hostname: u.hostname,
        port: u.port || (u.protocol === "https:" ? 443 : 80),
        method: "POST",
        path: pathWithQuery,
        agent: getAgent(u),
        headers: {
            "Content-Type": "application/octet-stream",
            "Content-Length": payloadBuf.byteLength,
            "Accept": "application/octet-stream"
        },
    };
    return new Promise((resolve) => {
        const t0 = performance.now();

        const req = mod.request(options, (res) => {
            let bytes = 0;
            res.on("data", (c) => {
                bytes += c.length;
            });
            res.on("end", () => {
                const t1 = performance.now();
                resolve({
                    ok: res.statusCode === 200,
                    status: res.statusCode,
                    latencyMs: t1 - t0,
                    //timing headers set by server.js
                    serverMs: res.headers["server-ms"] ? Number(res.headers["server-ms"]) : null,
                    computeMs: res.headers["compute-ms"] ? Number(res.headers["compute-ms"]) : null,
                    queueWaitMs: res.headers["queue-wait-ms"] ? Number(res.headers["queue-wait-ms"]) : null
                });
            });
        });
        //resolve with error info instead of rejecting to keep latency sample intact
        req.on("error", (e) => {
            const t1 = performance.now();
            resolve({
                ok: false,
                status: 0,
                latencyMs: t1 - t0,
                serverMs: null,
                computeMs: null,
                queueWaitMs: null,
                error: e.message
            });
        });

        req.write(payloadBuf);
        req.end();
    });
}
//single worker loop: sends sequential requests until deadline - return metrics.
async function runFixedConcurrency({
    serverBase,
    variant,
    workload,
    poolSize,
    payloadBuf,
    durationSec
}) {
    const deadline = performance.now() + durationSec * 1000;

    const latencies = [];
    const serverTimes = [];
    const computeTimes = [];
    const queueWaits = [];
    let ok = 0;
    let err = 0;

    const q = new URLSearchParams({
        width: String(workload.width),
        height: String(workload.height),
        kernelSize: String(workload.kernelSize),
        passes: String(workload.passes)
    });
    if (poolSize != null) {
        q.set("poolSize", String(poolSize));
    }
    const pathWithQuery = `/blur/${variant}?${q.toString()}`;

    while (performance.now() < deadline) {
        const r = await httpPostBinary(serverBase, pathWithQuery, payloadBuf);
        latencies.push(r.latencyMs);
        if (r.serverMs != null) serverTimes.push(r.serverMs);
        if (r.computeMs != null) computeTimes.push(r.computeMs);
        if (r.queueWaitMs != null && r.queueWaitMs > 0) queueWaits.push(r.queueWaitMs);

        if (r.ok) {
            ok++;
        } else {
            err++;
        }
    }

    return {
        latencies,
        serverTimes,
        computeTimes,
        queueWaits,
        ok,
        err
    };
}
//runs one full test case - warmup -> measure -> collect server metrics -> save JSON
async function runOneCase({
    outDir,
    serverBase,
    variant,
    workload,
    concurrency,
    poolSize,
    durationSec,
    warmupSec,
    seed,
    rep,
    tag
}) {
    await httpPostJson(serverBase, "/metrics/reset");
    const rgba = TestImageGenerator.generate(workload.width, workload.height, seed);
    const payloadBuf = Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength);

    const before = await httpGetJson(serverBase, "/metrics");
    //warmup
    if (warmupSec > 0) {
        const warmDeadline = performance.now() + warmupSec * 1000;
        const q = new URLSearchParams({
            width: String(workload.width),
            height: String(workload.height),
            kernelSize: String(workload.kernelSize),
            passes: String(workload.passes)
        });
        if (poolSize != null) {
            q.set("poolSize", String(poolSize));
        }
        const pathWithQuery = `/blur/${variant}?${q.toString()}`;

        async function warmWorker() {
            while (performance.now() < warmDeadline) {
                await httpPostBinary(serverBase, pathWithQuery, payloadBuf);
            }
        }
        await Promise.all(Array.from({ length: concurrency }, warmWorker));
    }
    //measured load phase
    const tStart = performance.now();
    const runners = Array.from({ length: concurrency }, () =>
        runFixedConcurrency({ serverBase, variant, workload, poolSize, payloadBuf, durationSec })
    );
    const results = await Promise.all(runners);
    const tEnd = performance.now();

    const after = await httpGetJson(serverBase, "/metrics");
    //merge results from worker slots
    const allLat = [];
    const allServerTimes = [];
    const allComputeTimes = [];
    const allQueueWaits = [];
    let ok = 0, err = 0;

    for (const r of results) {
        ok += r.ok;
        err += r.err;
        allLat.push(...r.latencies);
        allServerTimes.push(...r.serverTimes);
        allComputeTimes.push(...r.computeTimes);
        allQueueWaits.push(...r.queueWaits);
    }

    const wallSec = (tEnd - tStart) / 1000;
    const rps = wallSec > 0 ? ok / wallSec : 0;
    const errorRate = (ok + err) > 0 ? (err / (ok + err)) * 100 : 0;

    const out = {
        meta: {
            timestamp: new Date().toISOString(),
            tag,
            rep,
            serverBase,
            client: {
                platform: process.platform,
                arch: process.arch,
                node: process.version,
                cpus: os.cpus().length
            }
        },
        cfg: {
            variant,
            poolSize: poolSize ?? null,
            workload: {
                name: workload.name,
                width: workload.width,
                height: workload.height,
                kernelSize: workload.kernelSize,
                passes: workload.passes,
                payloadBytes: payloadBuf.byteLength
            },
            load: {
                concurrency,
                warmupSec,
                durationSec
            }
        },
        client: {
            requests: {
                ok,
                err,
                errorRatePercent: errorRate
            },
            wallSec,
            throughputRps: rps,
            latency: {
                client: summarizeLatencies(allLat),
                server: summarizeLatencies(allServerTimes),
                compute: summarizeLatencies(allComputeTimes),
                queueWait: summarizeLatencies(allQueueWaits)
            }
        },
        server: {
            before: {
                memory: before.memory,
                cpu: before.cpu,
                eventLoopLag: before.eventLoopLag
            },
            after: {
                requests: after.requests,
                memory: after.memory,
                cpu: after.cpu,
                eventLoopLag: after.eventLoopLag,
                latency: after.latency,
                inflight: after.inflight,
                queues: after.queues
            }
        }
    };
    //sanitize variant, concurrency and workload names for safe filenames
    const safe = (s) => String(s).replace(/[^\w.-]+/g, "_");
    const fname = `sys_${safe(variant)}_ps${poolSize ?? "na"}_${safe(workload.name)}_c${concurrency}_${safe(tag)}.json`;
    const fpath = path.join(outDir, fname);
    fs.writeFileSync(fpath, JSON.stringify(out, null, 2), "utf8");
    console.log(`[run] wrote ${fpath}`);
    console.log(`[run] ok=${ok} err=${err} rps=${rps.toFixed(2)} errorRate=${errorRate.toFixed(2)}% p95=${out.client.latency.client.p95.toFixed(2)}ms`);
}
//workloads matching compute benchmark
const COMPUTE_WORKLOADS = [
    { name: "small_512x512_k5_p1", width: 512, height: 512, kernelSize: 5, passes: 1 },
    { name: "medium_1920x1080_k15_p1", width: 1920, height: 1080, kernelSize: 15, passes: 1 },
    { name: "large_1920x1080_k15_p10", width: 1920, height: 1080, kernelSize: 15, passes: 10 },
    { name: "heavy_2560x1440_k31_p5", width: 2560, height: 1440, kernelSize: 31, passes: 5 }
];

//System-Benchmark configurations -- system_local = full benchmark, quick = quick test
const PRESETS = {
    system_local: {
        warmupSec: 2,
        durationSec: 10,
        reps: 1,
        concurrencies: [ 1, /* 2, */ 4, 8, 16/* , 32 */],
        poolSizes: [1, 4, 8, 16],
        variants: ["js_pure", "js_wasm", "js_wasm_simd", "js_worker", "js_worker_wasm", "js_worker_wasm_simd"],
        workloads: COMPUTE_WORKLOADS
    },
    quick: {
        warmupSec: 1,
        durationSec: 5,
        reps: 1,
        concurrencies: [4, 16],
        poolSizes: [8],
        variants: ["js_pure", "js_worker"],
        workloads: [COMPUTE_WORKLOADS[0]]
    },
};
//check for worker variants
function isWorkerVariant(v) {
    return v === "js_worker" || v === "js_worker_wasm" || v === "js_worker_wasm_simd";
}

async function main() {
    const args = parseArgs(process.argv);

    const serverBase = args.server || "http://localhost:3000";
    const presetName = args.preset || "system_local";
    const preset = PRESETS[presetName];
    if (!preset) {
        throw new Error(`Unknown preset: ${presetName}`);
    }

    const outDir = args.outDir || path.join(process.cwd(), "results_system");
    fs.mkdirSync(outDir, { recursive: true });

    const seed = args.seed != null ? toInt(args.seed, "seed") : 42;
    const tag = args.tag || presetName;
    const cooldownMs = args.cooldownMs != null ? toInt(args.cooldownMs, "cooldownMs") : 2000;

    console.log(`[loadgen] server=${serverBase}`);
    console.log(`[loadgen] preset=${presetName} tag=${tag}`);
    console.log(`[loadgen] outDir=${outDir}`);
    console.log(`[loadgen] warmupSec=${preset.warmupSec} durationSec=${preset.durationSec} reps=${preset.reps}`);
    console.log(`[loadgen] concurrencies=${JSON.stringify(preset.concurrencies)} poolSizes=${JSON.stringify(preset.poolSizes)}`);
    console.log(`[loadgen] variants=${JSON.stringify(preset.variants)}`);
    console.log(`[loadgen] workloads=${preset.workloads.map(w => w.name).join(", ")}\n`);

    await httpGetJson(serverBase, "/health");

    for (const workload of preset.workloads) {
        console.log(`\n### WORKLOAD ${workload.name} (${workload.width}x${workload.height} k${workload.kernelSize} p${workload.passes}) ###`);

        for (const variant of preset.variants) {
            //non-worker variants skip pool size loop
            const poolSizes = isWorkerVariant(variant) ? preset.poolSizes : [null];

            for (const poolSize of poolSizes) {
                for (const concurrency of preset.concurrencies) {
                    for (let rep = 1; rep <= preset.reps; rep++) {
                        console.log(`\n=== RUN variant=${variant} poolSize=${poolSize ?? "na"} conc=${concurrency} rep=${rep}/${preset.reps} ===`);

                        await runOneCase({
                            outDir,
                            serverBase,
                            variant,
                            workload,
                            concurrency,
                            poolSize,
                            durationSec: preset.durationSec,
                            warmupSec: preset.warmupSec,
                            seed,
                            rep,
                            tag
                        });

                        if (cooldownMs > 0) {
                            await sleep(cooldownMs);
                        }
                    }
                }
            }
        }
    }

    console.log("\n[loadgen] done.");
}

main().catch((e) => {
    console.error("[loadgen] error:", e && e.stack ? e.stack : e);
    process.exit(1);
});