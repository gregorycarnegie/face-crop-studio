import package::pixel::{unpack_pixel, pack_pixel};

const MASK_SMOOTH_INNER : f32 = 0.9;
const MASK_SMOOTH_OUTER : f32 = 1.1;

struct BackgroundBlurUniforms {
    width : u32,
    height : u32,
    mask_size : f32,
    _pad : f32,
};

@group(0) @binding(0)
var<storage, read> sharp_pixels : array<u32>;

@group(0) @binding(1)
var<storage, read> blur_pixels : array<u32>;

@group(0) @binding(2)
var<storage, read_write> output_pixels : array<u32>;

@group(0) @binding(3)
var<uniform> params : BackgroundBlurUniforms;

fn store_pixel(idx : u32, value : vec4<f32>) {
    output_pixels[idx] = pack_pixel(value);
}

fn compute_blend(px : vec2<f32>, size : vec2<f32>) -> f32 {
    let dist = length(px / size);
    let smooth_range = (dist - MASK_SMOOTH_INNER) / (MASK_SMOOTH_OUTER - MASK_SMOOTH_INNER);
    return clamp(smooth_range, 0.0, 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let sharp = vec2<f32>(f32(x), f32(y));
    let half = vec2<f32>(f32(params.width) * 0.5, f32(params.height) * 0.5);
    let radii = half * params.mask_size;
    let blend = compute_blend(sharp - half, radii);

    let sharp_px = unpack_pixel(sharp_pixels[idx]);
    let blur_px = unpack_pixel(blur_pixels[idx]);
    let color = mix(sharp_px, blur_px, vec4<f32>(blend, blend, blend, blend));
    store_pixel(idx, color);
}
