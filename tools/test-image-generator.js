//Linear congruential generator for deterministic pixeldata
class LCG {
    static a = 1664525;
    static c = 1013904223;
    static m = 2 ** 32;

    constructor(seed) {
        this.state = seed >>> 0;
    }

    nextInt() {
        this.state = (LCG.a * this.state + LCG.c) % LCG.m;
        return this.state;
    }
}

class TestImageGenerator {
    //generates deterministic RGBA image for a given seed
    static generate(width, height, seed = 42) {
        if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
            throw new RangeError("width and height must be positive integers");
        }

        const pixelData = new Uint8Array(width * height * 4);
        const generator = new LCG(seed);

        for (let i = 0; i < pixelData.length; i = i + 4) {
            pixelData[i] = generator.nextInt() & 255;
            pixelData[i + 1] = generator.nextInt() & 255;
            pixelData[i + 2] = generator.nextInt() & 255;
            pixelData[i + 3] = generator.nextInt() & 255;
        }
        return pixelData;
    }

    //only for visualization - not used
    static async savePNG(pixelData, width, height, filename) {
        const { createCanvas } = require("canvas");
        const fs = require("fs");

        const canvas = createCanvas(width, height);
        const context = canvas.getContext('2d');
        const imageData = context.createImageData(width, height);

        imageData.data.set(pixelData);
        context.putImageData(imageData, 0, 0);

        const buffer = canvas.toBuffer('image/png');
        fs.writeFileSync(filename, buffer);
    }
}

module.exports = { LCG, TestImageGenerator };