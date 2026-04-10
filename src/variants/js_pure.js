const GaussianBlur = require("../gaussian_blur");
const { performance } = require("perf_hooks");
//inputvalidation
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
//cache blur instance per kernelsize
const blurCache = new Map();
//returns cached blur instance or creates new
function getBlur(kernelSize) {
    let b = blurCache.get(kernelSize);
    if (!b) {
        b = new GaussianBlur(kernelSize);
        blurCache.set(kernelSize, b);
    }
    return b;
}

module.exports = {
    name: "js_pure",
    isWorker: false,

    async run(input, width, height, kernelSize, opts = {}) {
        const passes = opts.passes || 1;
        //input validation
        validateRGBA(input, width, height, kernelSize);

        const blur = getBlur(kernelSize);
        //compute time measurement
        const t0 = performance.now();
        const out = blur.apply(input, width, height, passes);
        const t1 = performance.now();
        const computeMs = t1 - t0;
        //output validation
        if (!(out instanceof Uint8Array)) {
            throw new TypeError("js_pure output must be of type: Uint8Array");
        }
        if (out.length !== input.length) {
            throw new RangeError("js_pure output length must be equal to input length");
        }
        //return timings if enabled
        if (opts.returnTimings) {
            return { out, timings: { computeMs } };
        }

        return out;
    }
};