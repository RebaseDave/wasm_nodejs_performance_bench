const express = require("express");
const http = require("http");
const { performance, monitorEventLoopDelay } = require("perf_hooks");

const jsPure = require("./variants/js_pure");
const jsWasm = require("./variants/js_wasm");
const jsWasmSimd = require("./variants/js_wasm_simd");
const jsWorker = require("./variants/js_worker");
const jsWorkerWasm = require("./variants/js_worker_wasm");
const jsWorkerWasmSimd = require("./variants/js_worker_wasm_simd");

const VARIANTS = new Map([
    ["js_pure", jsPure],
    ["js_wasm", jsWasm],
    ["js_wasm_simd", jsWasmSimd],
    ["js_worker", jsWorker],
    ["js_worker_wasm", jsWorkerWasm],
    ["js_worker_wasm_simd", jsWorkerWasmSimd]
]);
//queues per pool for requests waiting for free worker slot
const pendingQueues = new Map();

function getQueue(key) {
    if (!pendingQueues.has(key)) {
        pendingQueues.set(key, []);
    }
    return pendingQueues.get(key);
}
//wait for free workerslot, returns queue time
async function acquireSlot(key, poolSize, isWorker) {
    const t0 = performance.now();

    if (!isWorker) {
        incInflight(key);
        return 0;
    }
    if (getInflight(key) < poolSize) {
        incInflight(key);
        return 0;
    }
    //queue request until a slot is released
    await new Promise((resolve) => {
        getQueue(key).push(resolve);
    });

    return performance.now() - t0;
}
//release slot and dispatch next queued request
function releaseSlot(key, isWorker) {
    if (!isWorker) {
        decInflight(key);
        return;
    }

    const queue = getQueue(key);
    if (queue.length > 0) {
        //hand slot to next waiter without decrementing
        const next = queue.shift();
        next();
    } else {
        decInflight(key);
    }
}

function parseIntParam(v, name) {
    const n = Number(v);
    if (!Number.isInteger(n)) {
        throw new TypeError(`${name} must be of type: integer`);
    }
    return n;
}

function mean(arr) {
    if (arr.length === 0) return 0;
    let s = 0;
    for (const x of arr) s += x;
    return s / arr.length;
}

function percentile(sortedArr, p) {
    if (sortedArr.length === 0) return 0;
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
    if (samples.length === 0) {
        return { mean: 0, p50: 0, p95: 0, p99: 0, max: 0 };
    }
    const sorted = [...samples].sort((a, b) => a - b);
    return {
        mean: mean(samples),
        p50: percentile(sorted, 0.5),
        p95: percentile(sorted, 0.95),
        p99: percentile(sorted, 0.99),
        max: sorted[sorted.length - 1]
    };
}

const metrics = {
    startMs: performance.now(),
    reqOk: 0,
    reqErr: 0,
    cpuLast: process.cpuUsage(),
    cpuLastMs: performance.now(),
    latencies: [],
    queueWaits: [],
    computeTimes: [],
    maxSamples: 10000,
    peakMemory: {
        rss: 0,
        heapUsed: 0
    }
}
//elu monitor
const eld = monitorEventLoopDelay({ resolution: 10 });
eld.enable();
//inflight request counters per pool key and global
const inflight = {
    total: 0,
    byPool: new Map(),
    peakTotal: 0,
    peakByPool: new Map()
}
//poolkey uniquely identifies a worker pool
function inflightKey(variantName, poolSize) {
    return `${variantName}:${poolSize}`;
}

function getInflight(key) {
    return inflight.byPool.get(key) || 0;
}

function incInflight(key) {
    inflight.total++;
    inflight.peakTotal = Math.max(inflight.peakTotal, inflight.total);

    const v = (inflight.byPool.get(key) || 0) + 1;
    inflight.byPool.set(key, v);

    inflight.peakByPool.set(key, Math.max(inflight.peakByPool.get(key) || 0, v));
}

function decInflight(key) {
    inflight.total = Math.max(0, inflight.total - 1);
    const v = getInflight(key) - 1;
    if (v <= 0) {
        inflight.byPool.delete(key);
    }
    else {
        inflight.byPool.set(key, v);
    }
}
//sample recorders - drop oldest entry when samples capped
function recordLatency(latencyMs) {
    metrics.latencies.push(latencyMs);
    if (metrics.latencies.length > metrics.maxSamples) {
        metrics.latencies.shift();
    }
}

function recordQueueWait(queueWaitMs) {
    if (queueWaitMs > 0) {
        metrics.queueWaits.push(queueWaitMs);
        if (metrics.queueWaits.length > metrics.maxSamples) {
            metrics.queueWaits.shift();
        }
    }
}

function recordComputeTime(computeMs) {
    if (computeMs > 0) {
        metrics.computeTimes.push(computeMs);
        if (metrics.computeTimes.length > metrics.maxSamples) {
            metrics.computeTimes.shift();
        }
    }
}
//periodic sampling for memoryusage
function updatePeakMemory() {
    const mem = process.memoryUsage();
    metrics.peakMemory.rss = Math.max(metrics.peakMemory.rss, mem.rss);
    metrics.peakMemory.heapUsed = Math.max(metrics.peakMemory.heapUsed, mem.heapUsed);
}

setInterval(updatePeakMemory, 200);
//read elu percentiles
function snapshotEventLoop() {
    const nsToMs = (ns) => ns / 1e6;

    return {
        p95: nsToMs(eld.percentile(95)),
        p99: nsToMs(eld.percentile(99)),
        max: nsToMs(eld.max)
    };
}
//calc cpu usage since last call
function snapshotCpuPercentSinceLast() {
    const nowMs = performance.now();
    const cpuNow = process.cpuUsage(metrics.cpuLast);

    const wallMs = nowMs - metrics.cpuLastMs;
    metrics.cpuLastMs = nowMs;
    metrics.cpuLast = process.cpuUsage();

    const cpuMs = (cpuNow.user + cpuNow.system) / 1000;
    const cpuPercent = wallMs > 0 ? (cpuMs / wallMs) * 100 : 0;

    return { cpuPercent };
}

const app = express();

app.get("/health", (req, res) => res.json({ ok: true }));

app.get("/metrics", (req, res) => {
    const upMs = performance.now() - metrics.startMs;
    const upSec = upMs / 1000;
    const mem = process.memoryUsage();
    const cpu = snapshotCpuPercentSinceLast();
    const ev = snapshotEventLoop();

    const totalReqs = metrics.reqOk + metrics.reqErr;
    const rps = upSec > 0 ? totalReqs / upSec : 0;
    const errorRate = totalReqs > 0 ? (metrics.reqErr / totalReqs) * 100 : 0;

    res.json({
        uptime: {
            ms: upMs,
            sec: upSec
        },
        requests: {
            total: totalReqs,
            ok: metrics.reqOk,
            err: metrics.reqErr,
            errorRatePercent: errorRate,
            rps: rps
        },
        latency: {
            server: statsCompact(metrics.latencies),
            queueWait: statsCompact(metrics.queueWaits),
            compute: statsCompact(metrics.computeTimes)
        },
        cpu,
        eventLoopLag: ev,
        memory: {
            current: {
                rss: mem.rss,
                heapUsed: mem.heapUsed
            },
            peak: metrics.peakMemory
        },
        inflight: {
            current: inflight.total,
            peak: inflight.peakTotal
        },
        queues: {
            pending: Array.from(pendingQueues.values()).reduce((s, q) => s + q.length, 0)
        }
    });
});

// reset all metrics and counters
app.post("/metrics/reset", (req, res) => {
    eld.reset();
    metrics.cpuLast = process.cpuUsage();
    metrics.cpuLastMs = performance.now();

    metrics.startMs = performance.now();
    metrics.reqOk = 0;
    metrics.reqErr = 0;
    metrics.latencies = [];
    metrics.queueWaits = [];
    metrics.computeTimes = [];

    const mem = process.memoryUsage();
    metrics.peakMemory.rss = mem.rss;
    metrics.peakMemory.heapUsed = mem.heapUsed;

    inflight.peakTotal = inflight.total;
    inflight.peakByPool.clear();

    if (inflight.total === 0) {
        inflight.byPool.clear();
        pendingQueues.clear();
    }

    res.json({ ok: true });
});
//main blur endpoint - streams raw rgba body, runs selected variant, returns rgba
app.post("/blur/:variant", async (req, res) => {
    const t0 = performance.now();
    let key = null;
    let acquired = false;
    let selectedVariant = null;
    let queueWaitMs = 0;

    try {
        const width = parseIntParam(req.query.width, "width");
        const height = parseIntParam(req.query.height, "height");
        const expectedLength = width * height * 4;
        //stream raw rgba body into buffer
        const cur = new Uint8Array(expectedLength);
        let offset = 0;

        for await (const chunk of req) {
            if (offset + chunk.length > expectedLength) {
                throw new Error(`Payload larger than expectedLength`);
            }
            cur.set(chunk, offset);
            offset += chunk.length;
        }
        if (offset !== expectedLength) {
            throw new Error(`Offset must be equal to expectedLength`);
        }

        const variantName = req.params.variant;
        selectedVariant = VARIANTS.get(variantName);
        if (!selectedVariant) {
            metrics.reqErr++;
            return res.status(404).json({ error: `Unknown variant: ${variantName}` });
        }
        //poolsize
        const poolSize = req.query.poolSize != null
            ? parseIntParam(req.query.poolSize, "poolSize")
            : (process.env.POOL_SIZE ? parseIntParam(process.env.POOL_SIZE, "POOL_SIZE") : 4);

        key = inflightKey(variantName, poolSize);
        queueWaitMs = await acquireSlot(key, poolSize, selectedVariant.isWorker);
        acquired = true;

        const opts = {
            size: poolSize,
            enforceTransfer: true,
            passes: parseIntParam(req.query.passes ?? 1, "passes"),
            returnTimings: true
        };

        const result = await selectedVariant.run(cur, width, height, parseIntParam(req.query.kernelSize, "kernelSize"), opts);
        const finalU8 = result.out ? result.out : result;
        const timings = result.timings || {};

        if (timings.computeMs) {
            res.setHeader("compute-ms", timings.computeMs.toFixed(3));
            recordComputeTime(timings.computeMs);
        }
        //wrap uint8 in Buffer for express response
        const outBuf = Buffer.from(finalU8.buffer, finalU8.byteOffset, finalU8.byteLength);

        metrics.reqOk++;

        const serverMs = performance.now() - t0;
        res.setHeader("server-ms", serverMs.toFixed(3));
        if (queueWaitMs > 0) {
            res.setHeader("queue-wait-ms", queueWaitMs.toFixed(3));
        }

        recordLatency(serverMs);
        recordQueueWait(queueWaitMs);

        res.status(200).type("application/octet-stream").send(outBuf);
    } catch (err) {
        metrics.reqErr++;
        const t1 = performance.now();
        res.setHeader("server-ms", (t1 - t0).toFixed(3));

        res.status(400).json({
            error: err && err.message ? err.message : String(err),
            name: err && err.name ? err.name : "Error"
        });

    } finally {
        //always release slot
        if (acquired && key) {
            releaseSlot(key, selectedVariant.isWorker);
        }
    }
});

const port = process.env.PORT ? Number(process.env.PORT) : 3000;
const server = http.createServer(app);
server.keepAliveTimeout = 65_000;
server.headersTimeout = 66_000;
server.maxRequestsPerSocket = 0;
server.listen(port, "0.0.0.0", () => {
    console.log(`[server] listening on http://0.0.0.0:${port}`);
    console.log(`[server] POST /blur/:variant?width=&height=&kernelSize=&passes=&poolSize=`);
    console.log(`[server] GET /metrics`);
    console.log(`[server] POST /metrics/reset`);
});