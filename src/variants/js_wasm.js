const { performance } = require("perf_hooks");

//lazy-load wasm module on first use
let wasmMod = null;
function getWasm() {
    if (wasmMod) {
        return wasmMod;
    }
    wasmMod = require("../../gaussian_blur_wasm/pkg");
    return wasmMod;
}
//input validation
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
        throw new RangeError("input does not match requirements: width * height * 4");
    }
}

module.exports = {
    name: "js_wasm",
    isWorker: false,

    async run(input, width, height, kernelSize, opts = {}) {
        validateRGBA(input, width, height, kernelSize);
        const passes = opts.passes || 1;

        const wasm = getWasm();
        //run wasm blur and measure isolated compute-time
        const t0 = performance.now();
        const current = wasm.gaussian_blur_rgba(input, width, height, kernelSize, passes);
        const t1 = performance.now();

        const computeMs = t1 - t0;

        const out = current;
        //output validation
        if (!(out instanceof Uint8Array)) {
            throw new TypeError("js_wasm output must be of type: Uint8Array");
        }
        if (out.length !== input.length) {
            throw new RangeError("js_wasm output length must be equal to input length");
        }
        //return timings
        if (opts.returnTimings) {
            return { out, timings: { computeMs } };
        }

        return out;
    }
};