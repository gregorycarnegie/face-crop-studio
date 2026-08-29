struct PreprocessUniforms {
    src_size : vec2<u32>,
    dst_size : vec2<u32>,
};

@group(0) @binding(0)
var source_tex : texture_2d<f32>;

@group(0) @binding(1)
var source_sampler : sampler;

@group(0) @binding(2)
var<storage, read_write> output_buffer : array<f32>;

@group(0) @binding(3)
var<uniform> uniforms : PreprocessUniforms;

// Upper bound on taps per axis. At the cap the box is undersampled again, but a 16x
// downscale already averages ~1024 source pixels per output pixel through 256 taps, and
// the cost is quadratic — this bounds the worst case rather than chasing exactness.
const MAX_TAPS : u32 = 16u;

fn chw_index(x : u32, y : u32, c : u32, width : u32, height : u32) -> u32 {
    return c * width * height + y * width + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= uniforms.dst_size.x || global_id.y >= uniforms.dst_size.y) {
        return;
    }

    let src_size_f = vec2<f32>(vec2<u32>(uniforms.src_size));
    let dst_size_f = vec2<f32>(vec2<u32>(uniforms.dst_size));

    // Source texels covered by one destination texel. A single bilinear tap only ever blends
    // a 2x2 neighbourhood, so on minification it samples 4 source pixels out of `ratio.x *
    // ratio.y` and the result aliases: a 9x downscale was reading 4 of ~54 pixels, which moved
    // real detections (23px on a landmark) rather than merely losing precision. Average enough
    // taps to cover the box instead. Each tap is itself bilinear and so spans ~2 texels, hence
    // the halved tap count.
    let ratio = src_size_f / dst_size_f;
    let taps = clamp(
        vec2<u32>(ceil(ratio * 0.5)),
        vec2<u32>(1u, 1u),
        vec2<u32>(MAX_TAPS, MAX_TAPS),
    );
    let taps_f = vec2<f32>(taps);

    // With taps == 1 the single sample lands at (global_id + 0.5) / dst_size, which is exactly
    // the previous formula — magnification and the no-resize case are unchanged.
    var accum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var j : u32 = 0u; j < taps.y; j = j + 1u) {
        for (var i : u32 = 0u; i < taps.x; i = i + 1u) {
            let offset = (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5, 0.5)) / taps_f;
            let src_pos = (vec2<f32>(global_id.xy) + offset) * ratio;
            accum = accum + textureSampleLevel(source_tex, source_sampler, src_pos / src_size_f, 0.0);
        }
    }
    let color = accum / (taps_f.x * taps_f.y);

    let width = uniforms.dst_size.x;
    let height = uniforms.dst_size.y;
    let idx = chw_index(global_id.x, global_id.y, 0u, width, height);
    let plane_size = width * height;

    // Convert from normalized [0,1] floats back to 0-255 range and swap RGB->BGR.
    output_buffer[idx] = color.b * 255.0;
    output_buffer[idx + plane_size] = color.g * 255.0;
    output_buffer[idx + plane_size * 2u] = color.r * 255.0;
}
