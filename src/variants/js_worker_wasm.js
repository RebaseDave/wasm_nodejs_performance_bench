//same as js_worker but uses blur_worker_wasm

const path = require("path");
const WorkerPool = require("./worker_pool");
const workerFile = path.join(__dirname, "workers", "blur_worker_wasm.js");

function validateRGBA(input, width, height, kernelSize) {
    if (!(input instanceof Uint8Array)) {
        throw new TypeError("input must be of type: Uint8Array");
    }
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
        throw new RangeError("width and height must be > 0");
    }
    if (!Number.isInteger(kernelSize) || kernelSize <= 0 || kernelSize % 2 == 0) {
        throw new RangeError("kernelSize must be positive integer and odd");
    }
    if (input.length !== width * height * 4) {
        throw new RangeError("input does not match requirements: width * height * 4")
    }
    if (input.byteOffset !== 0 || input.byteLength !== input.buffer.byteLength) {
        throw new TypeError("input must cover full ArrayBuffer");
    }
}

const pools = new Map();

function getPool(opts = {}) {
    const size = Number.isInteger(opts.size) && opts.size > 0 ? opts.size : 4;
    const enforceTransfer = opts.enforceTransfer !== false;
    const key = `${size}:${enforceTransfer ? 1 : 0}`;
    let pool = pools.get(key);

    if (!pool) {
        pool = new WorkerPool({ workerFile, size, enforceTransfer });
        pools.set(key, pool);
    }
    return pool;
}

module.exports = {
    name: "js_worker_wasm",
    isWorker: true,

    async run(input, width, height, kernelSize, opts = {}) {
        validateRGBA(input, width, height, kernelSize);

        const p = getPool(opts);

        const payload = { input: input.buffer, width, height, kernelSize, passes: opts.passes || 1, returnTimings: opts.returnTimings === true };

        const res = await p.run(payload, [payload.input]);

        if (res instanceof Uint8Array || (res && res.out instanceof Uint8Array)) {
            return res;
        }
        if (res && typeof res === "object") {
            return res;
        }
        throw new TypeError("js_worker_wasm output must be of type: Uint8Array or { out: Uint8Array, timings }");
    },

    async closeAll() {
        const all = [...pools.values()];
        pools.clear();
        await Promise.all(all.map(pool => pool.close()));
    }
};