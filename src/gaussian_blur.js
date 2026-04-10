class GaussianBlur {
    constructor(kernelSize) {
        this.kernelSize = kernelSize;
        this.sigma = kernelSize / 3;
        this.kernel = this.createKernel();
    }
    //build normalized 1D gaussian kernel
    createKernel() {
        const kernelMatrix = new Float32Array(this.kernelSize);
        const center = Math.floor(this.kernelSize / 2);
        let kernelSum = 0;

        for (let i = 0; i < this.kernelSize; i++) {
            let x = i - center;
            kernelMatrix[i] = Math.exp(-(x * x) / (2 * this.sigma * this.sigma));
            kernelSum += kernelMatrix[i];
        }

        for (let i = 0; i < this.kernelSize; i++) {
            kernelMatrix[i] = kernelMatrix[i] / kernelSum;
        }

        return kernelMatrix;
    }
    //apply separable gaussian blur for n passes
    apply(imageData, width, height, passes = 1) {
        if (!Number.isInteger(this.kernelSize) || this.kernelSize <= 0 || this.kernelSize % 2 === 0) {
            throw new Error("kernelSize must be positive and odd");
        }
        if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
            throw new Error("width and height must be > 0");
        }
        if (!imageData || imageData.length !== width * height * 4) {
            throw new Error("input length does not match requirements: width * height * 4");
        }

        const input = imageData instanceof Uint8Array
            ? imageData
            : new Uint8Array(imageData.buffer, imageData.byteOffset ?? 0, imageData.byteLength ?? imageData.length);
        //ping-pong buffers
        const buf1 = new Uint8Array(input.length);
        const buf2 = new Uint8Array(input.length);
        const temp = new Uint8Array(input.length);
        buf1.set(input);

        for(let p = 0; p < passes; p++){
            const src = p % 2 === 0 ? buf1 : buf2;
            const dst = p % 2 === 0 ? buf2 : buf1;
            this.convolveHorizontal(src, temp, width, height);
            this.convolveVertical(temp, dst, width, height);
        }
        //return the buffer that received the last write
        return passes % 2 === 0 ? buf1 : buf2;
    }
    //horizontal 1D convolution pass
    convolveHorizontal(input, output, width, height) {
        const kernelRadius = Math.floor(this.kernelSize / 2);
        const kernel = this.kernel;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let r = 0, g = 0, b = 0, a = 0;

                for (let k = -kernelRadius; k <= kernelRadius; k++) {
                    const xx = this.clampIndex(x + k, 0, width - 1);    //clamp at image border
                    const idx = (y * width + xx) * 4;
                    const w = kernel[k + kernelRadius];

                    r += input[idx] * w;
                    g += input[idx + 1] * w;
                    b += input[idx + 2] * w;
                    a += input[idx + 3] * w;
                }

                const outIdx = (y * width + x) * 4;
                output[outIdx] = this.clampByte(r);
                output[outIdx + 1] = this.clampByte(g);;
                output[outIdx + 2] = this.clampByte(b);;
                output[outIdx + 3] = this.clampByte(a);;
            }
        }
    }
    //vertical 1D convolution pass
    convolveVertical(input, output, width, height) {
        const kernelRadius = Math.floor(this.kernelSize / 2);
        const kernel = this.kernel;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let r = 0, g = 0, b = 0, a = 0;

                for (let k = -kernelRadius; k <= kernelRadius; k++) {
                    const yy = this.clampIndex(y + k, 0, height - 1);
                    const idx = (yy * width + x) * 4;
                    const w = kernel[k + kernelRadius];

                    r += input[idx] * w;
                    g += input[idx + 1] * w;
                    b += input[idx + 2] * w;
                    a += input[idx + 3] * w;
                }

                const outIdx = (y * width + x) * 4;
                output[outIdx] = this.clampByte(r);;
                output[outIdx + 1] = this.clampByte(g);;
                output[outIdx + 2] = this.clampByte(b);;
                output[outIdx + 3] = this.clampByte(a);;
            }
        }
    }
    //clamp pixel coordinate to image range
    clampIndex(value, min, max) {
        if (value < min) {
            return min;
        }
        if (value > max) {
            return max;
        }
        else {
            return value;
        }
    }
    //round and clamp float to valid byte range
    clampByte(value) {
        const rounded = Math.round(value);
        if (rounded < 0) {
            return 0;
        }
        if (rounded > 255) {
            return 255;
        }
        else {
            return rounded;
        }
    }
}

module.exports = GaussianBlur;