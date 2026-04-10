use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

#[wasm_bindgen]
pub fn gaussian_blur_rgba_simd(
    input: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    passes: usize,
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

    #[cfg(not(target_arch = "wasm32"))]
{
    return Err(JsValue::from_str(
        "gaussian_blur_rgba_simd requires wasm32 + simd128",
    ));
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

        #[cfg(target_arch = "wasm32")]
        unsafe {
            convolve_horizontal_simd(src, &mut temp, width, height, &kernel);
            convolve_vertical_simd(&temp, dst, width, height, &kernel);
        }
    }

    Ok(if passes % 2 == 0 { buf1 } else { buf2 })
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn convolve_horizontal_simd(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    kernel: &[f32],
) {
    let radius = (kernel.len() / 2) as isize;

    for y in 0..height {
        for x in 0..width {
            let mut acc = f32x4_splat(0.0);

            for (k_idx, &weight) in kernel.iter().enumerate() {
                let k = k_idx as isize - radius;
                let xx = (x as isize + k).clamp(0, width as isize - 1) as usize;
                let idx = (y * width + xx) * 4;

                let pixel_f32 = f32x4(input[idx] as f32, input[idx + 1] as f32,
                                      input[idx + 2] as f32, input[idx + 3] as f32);

                let weight_vec = f32x4_splat(weight);
                acc = f32x4_add(acc, f32x4_mul(pixel_f32, weight_vec));
            }

            let out_idx = (y * width + x) * 4;
            output[out_idx]     = f32x4_extract_lane::<0>(acc).round().clamp(0.0, 255.0) as u8;
            output[out_idx + 1] = f32x4_extract_lane::<1>(acc).round().clamp(0.0, 255.0) as u8;
            output[out_idx + 2] = f32x4_extract_lane::<2>(acc).round().clamp(0.0, 255.0) as u8;
            output[out_idx + 3] = f32x4_extract_lane::<3>(acc).round().clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn convolve_vertical_simd(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    kernel: &[f32],
) {
    let radius = (kernel.len() / 2) as isize;

    for y in 0..height {
        for x in 0..width {
            let mut acc = f32x4_splat(0.0);

            for (k_idx, &weight) in kernel.iter().enumerate() {
                let k = k_idx as isize - radius;
                let yy = (y as isize + k).clamp(0, height as isize - 1) as usize;
                let idx = (yy * width + x) * 4;

                let pixel_f32 = f32x4(
                    input[idx] as f32,
                    input[idx + 1] as f32,
                    input[idx + 2] as f32,
                    input[idx + 3] as f32,
                );

                let weight_vec = f32x4_splat(weight);
                acc = f32x4_add(acc, f32x4_mul(pixel_f32, weight_vec));
            }

            let out_idx = (y * width + x) * 4;
            output[out_idx]     = f32x4_extract_lane::<0>(acc).round().clamp(0.0, 255.0) as u8;
            output[out_idx + 1] = f32x4_extract_lane::<1>(acc).round().clamp(0.0, 255.0) as u8;
            output[out_idx + 2] = f32x4_extract_lane::<2>(acc).round().clamp(0.0, 255.0) as u8;
            output[out_idx + 3] = f32x4_extract_lane::<3>(acc).round().clamp(0.0, 255.0) as u8;
        }
    }
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