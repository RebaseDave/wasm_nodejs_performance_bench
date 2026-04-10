use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn gaussian_blur_rgba(
    input: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    passes: usize
) -> Result<Vec<u8>, JsValue> {
    if kernel_size == 0 || kernel_size % 2 == 0 {
        return Err(JsValue::from_str("kernel_size must be positive and odd"));
    }
    if width == 0 || height == 0 {
        return Err(JsValue::from_str("width and height must be > 0"));
    }
    if input.len() != width * height * 4 {
        return Err(JsValue::from_str("input length does not match requirements: width * height * 4"));
    }
    if passes == 0 {
        return Err(JsValue::from_str("passes must be > 0"));
    }
    
    let kernel = create_kernel(kernel_size);

    let mut buf1 = input.to_vec();
    let mut buf2 = vec![0u8; input.len()];
    let mut temp = vec![0u8; input.len()];

    for p in 0..passes {
        let (src, dst) = if p % 2 == 0 {
            (&buf1[..], &mut buf2[..])
        } else {
            (&buf2[..], &mut buf1[..])
        };
        convolve_horizontal(src, &mut temp, width, height, &kernel);
        convolve_vertical(&temp, dst, width, height, &kernel);
    }

    Ok(if passes % 2 == 0 { buf1 } else { buf2 })
}


fn create_kernel(kernel_size: usize) -> Vec<f32> {
    let sigma = kernel_size as f32 / 3.0;
    let center = kernel_size as isize / 2;

    let mut kernel = vec![0.0f32; kernel_size];
    let mut sum = 0.0;

    for i in 0..kernel_size {
        let x = i as isize - center;
        let value = (-(x * x) as f32 / (2.0 * sigma * sigma)).exp();
        kernel[i] = value;
        sum += value;
    }

    for v in kernel.iter_mut() {
        *v /= sum;
    }

    kernel
}

fn convolve_horizontal(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    kernel: &[f32],
) {
    let radius = kernel.len() as isize / 2;

    for y in 0..height {
        for x in 0..width {
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut a = 0.0;

            for k in -radius..=radius {
                let xx = clamp(x as isize + k, 0, width as isize - 1) as usize;
                let idx = (y * width + xx) * 4;
                let w = kernel[(k + radius) as usize];

                r += input[idx] as f32 * w;
                g += input[idx + 1] as f32 * w;
                b += input[idx + 2] as f32 * w;
                a += input[idx + 3] as f32 * w;
            }

            let out_idx = (y * width + x) * 4;
            output[out_idx] = r.round().clamp(0.0, 255.0) as u8;
            output[out_idx + 1] = g.round().clamp(0.0, 255.0) as u8;
            output[out_idx + 2] = b.round().clamp(0.0, 255.0) as u8;
            output[out_idx + 3] = a.round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn convolve_vertical(input: &[u8], output: &mut [u8], width: usize, height: usize, kernel: &[f32]) {
    let radius = kernel.len() as isize / 2;

    for y in 0..height {
        for x in 0..width {
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut a = 0.0;

            for k in -radius..=radius {
                let yy = clamp(y as isize + k, 0, height as isize - 1) as usize;
                let idx = (yy * width + x) * 4;
                let w = kernel[(k + radius) as usize];

                r += input[idx] as f32 * w;
                g += input[idx + 1] as f32 * w;
                b += input[idx + 2] as f32 * w;
                a += input[idx + 3] as f32 * w;
            }

            let out_idx = (y * width + x) * 4;
            output[out_idx] = r.round().clamp(0.0, 255.0) as u8;
            output[out_idx + 1] = g.round().clamp(0.0, 255.0) as u8;
            output[out_idx + 2] = b.round().clamp(0.0, 255.0) as u8;
            output[out_idx + 3] = a.round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn clamp(value: isize, min: isize, max: isize) -> isize {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}
