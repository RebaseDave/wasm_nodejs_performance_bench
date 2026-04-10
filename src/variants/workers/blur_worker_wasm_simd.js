//same as blur_worker_wasm but uses simd function
const { parentPort } = require("worker_threads");
const { performance } = require("perf_hooks");
const wasmSimd = require("../../../gaussian_blur_wasm_simd/pkg");

const blurCache = new Map();

function getBlur(kernelSize) {
    let b = blurCache.get(kernelSize);
    if (!b) {
        b = {
            kernelSize, apply: (input, width, height, passes) => {
                return wasmSimd.gaussian_blur_rgba_simd(input, width, height, kernelSize, passes);
            }
        };
        blurCache.set(kernelSize, b);
    }
    return b;
}

if (!parentPort) {
    throw new Error("blur_worker_wasm_simd must be run as Worker");
}

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

        let currentU8 = new Uint8Array(msg.input);
        const expectedLength = width * height * 4;

        const blur = getBlur(kernelSize);

        const t0 = performance.now();

        currentU8 = blur.apply(currentU8, width, height, passes)
        const t1 = performance.now();

        const computeMs = t1 - t0;

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
        const resp = { jobId, ok: true, out: outBuf };

        if (msg.returnTimings === true) {
            resp.timings = {
                computeMs,
                queueWaitMs: Number.isFinite(msg._dispatchDelayMs) ? msg._dispatchDelayMs : null
            };
        }
        parentPort.postMessage(resp, [outBuf]);

    } catch (e) {
        const message = e && e.message ? e.message : String(e);
        parentPort.postMessage({ jobId: Number.isInteger(jobId) ? jobId : null, ok: false, error: message });
    }
});