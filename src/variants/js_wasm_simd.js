//same as js_wasm but uses simd function

const { performance } = require("perf_hooks");

let wasmSimdMod = null;

function getWasmSimd() {
    if (wasmSimdMod) {
        return wasmSimdMod;
    }

    wasmSimdMod = require("../../gaussian_blur_wasm_simd/pkg");
    return wasmSimdMod;
}

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
    name: "js_wasm_simd",
    isWorker: false,

    async run(input, width, height, kernelSize, opts = {}) {
        validateRGBA(input, width, height, kernelSize);
        const passes = opts.passes || 1;

        const wasmSimd = getWasmSimd();

        const t0 = performance.now();
        const out = wasmSimd.gaussian_blur_rgba_simd(input, width, height, kernelSize, passes);
        const t1 = performance.now();

        const computeMs = t1 - t0;

        if (!(out instanceof Uint8Array)) {
            throw new TypeError("js_wasm_simd output must be of type: Uint8Array");
        }
        if (out.length !== input.length) {
            throw new RangeError("js_wasm_simd output length must be equal to input length");
        }

        if (opts.returnTimings) {
            return { out, timings: { computeMs } };
        }

        return out;
    }
};