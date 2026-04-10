const { parentPort } = require("worker_threads");
const { performance } = require("perf_hooks");
const GaussianBlur = require("../../gaussian_blur");

//cache blur instance per kernel
const blurCache = new Map();

//return cached blur instance or creates new one
function getBlur(kernelSize) {
    let b = blurCache.get(kernelSize);
    if (!b) {
        b = new GaussianBlur(kernelSize);
        blurCache.set(kernelSize, b);
    }
    return b;
}

if (!parentPort) {
    throw new Error("blur_worker_js must be run as Worker");
}

//handle incoming blur jobs from workerpool
parentPort.on("message", (msg) => {
    const jobId = msg && msg.jobId;

    try {
        if (!msg || typeof msg !== "object") {
            throw new TypeError("message must be of type: object");
        }
        if (!Number.isInteger(jobId)) {
            throw new TypeError("jobId must be of type: integer");
        }
        if (!(msg.input instanceof ArrayBuffer)) {
            throw new TypeError("input must be of type: ArrayBuffer");
        }

        const { width, height, kernelSize, passes = 1 } = msg;
        const expectedLength = width * height * 4;
        //run and measure compute time
        const blur = getBlur(kernelSize);
        const t0 = performance.now();
        const currentU8 = blur.apply(new Uint8Array(msg.input), width, height, passes);
        const t1 = performance.now();
        const computeMs = t1 - t0;

        //output validation
        if (!(currentU8 instanceof Uint8Array)) {
            throw new TypeError("worker output must be of type: Uint8Array");
        }
        if (currentU8.byteLength !== expectedLength) {
            throw new RangeError("output byteLength must be equal to input byteLength")
        }
        if (currentU8.byteOffset !== 0 || currentU8.byteLength !== currentU8.buffer.byteLength) {
            throw new TypeError("worker output must cover full ArrayBuffer")
        }

        const outBuf = currentU8.buffer;
        if (outBuf.byteLength === 0) {
            throw new Error("Worker produced deteached ArrayBuffer");
        }
        const resp = { jobId, ok: true, out: outBuf };
        //attach timings
        if (msg.returnTimings === true) {
            resp.timings = {
                computeMs,
                queueWaitMs: typeof msg._dispatchDelayMs === "number" ? msg._dispatchDelayMs : null
            };
        }
        //transfer ArrayBuffer by transferList
        parentPort.postMessage(resp, [outBuf]);
    } catch (e) {
        const message = e && e.message ? e.message : String(e);
        parentPort.postMessage({ jobId: Number.isInteger(jobId) ? jobId : null, ok: false, error: message });
    }
});