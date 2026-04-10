const { TestImageGenerator } = require("../../tools/test-image-generator");

//builds single workload with pre-generated input images
function buildWorkload({ name, width, height, kernelSize, passes = 1, iterations, baseSeed }) {
    if (typeof name !== "string" || name.length === 0) {
        throw new TypeError("name must be non empty string");
    }
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
        throw new RangeError("width and height must be positive integers");
    }
    if (!Number.isInteger(kernelSize) || kernelSize <= 0 || kernelSize % 2 === 0) {
        throw new RangeError("kernel size must be positive integer and odd");
    }
    if (!Number.isInteger(iterations) || iterations <= 0) {
        throw new RangeError("iterations must be a positive integer");
    }
    if (!Number.isInteger(baseSeed)) {
        throw new TypeError("baseSeed must be of type: integer");
    }
    if (!Number.isInteger(passes) || passes <= 0) {
        throw new RangeError("passes must be a positive integer");
    }

    const inputs = new Array(iterations);
    const expectedBytes = width * height * 4;

    for (let i = 0; i < iterations; i++) {
        //generates image per iteration using offset seed
        const seed = (baseSeed + i) >>> 0;
        const input = TestImageGenerator.generate(width, height, seed);

        if (input.byteOffset !== 0 || input.byteLength !== input.buffer.byteLength) {
            throw new Error(`${name}: input must cover full ArrayBuffer`);
        }

        if (input.byteLength !== expectedBytes) {
            throw new Error(`${name}: expected ${expectedBytes} bytes, got ${input.byteLength}`);
        }

        inputs[i] = input;
    }
    return { name, width, height, kernelSize, passes, iterations, baseSeed, inputs };
}
//builds all workload definitions for benchmark run
function buildWorkloads(cfg = {}) {
    const iterations = Number.isInteger(cfg.iterations) && cfg.iterations > 0 ? cfg.iterations : 30;
    const baseSeed = Number.isInteger(cfg.baseSeed) ? (cfg.baseSeed | 0) : 42;

    const defs = [
        /* { name: "tiny_128x128_k3_p1", width: 128, height: 128, kernelSize: 3, passes: 1 }, */
        { name: "small_512x512_k5_p1", width: 512, height: 512, kernelSize: 5, passes: 1 },
        { name: "medium_1920x1080_k15_p1", width: 1920, height: 1080, kernelSize: 15, passes: 1 },
        { name: "large_1920x1080_k15_p10", width: 1920, height: 1080, kernelSize: 15, passes: 10 },
        { name: "heavy_2560x1440_k31_p5", width: 2560, height: 1440, kernelSize: 31, passes: 5},
        { name: "ultra_3840x2160_k31_p5", width: 3840, height: 2160, kernelSize: 31, passes: 5 }
    ];

    return defs.map((d, idx) => {
        //offset by prime to avoid seed collisions between workloads
        const wlSeed = ((baseSeed >>> 0) + idx * 10007) >>> 0;
        return buildWorkload({ ...d, iterations, baseSeed: wlSeed })
    });
}

module.exports = { buildWorkloads };