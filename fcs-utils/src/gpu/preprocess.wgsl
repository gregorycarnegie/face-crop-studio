struct PreprocessUniforms {
    src_size : vec2<u32>,
    dst_size : vec2<u32>,
    // Where the source is drawn inside the destination, and how large it is drawn. The host
    // fits the source without distorting it (see `fit_input`), so `drawn` keeps the source
    // aspect and everything outside it is padding.
    origin : vec2<u32>,
    drawn : vec2<u32>,
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

// Fill value for the letterbox bars, in the same 0-255 range as the pixels. Must match
// `fcs_utils::LETTERBOX_PAD`, which the CPU path writes and the parity tests compare.
const PAD : f32 = 0.0;

fn chw_index(x : u32, y : u32, c : u32, width : u32, height : u32) -> u32 {
    return c * width * height + y * width + x;
}

/// Write one destination pixel, swapping RGB->BGR on the way into the planar tensor.
fn write_pixel(pos : vec2<u32>, rgb : vec3<f32>) {
    let width = uniforms.dst_size.x;
    let height = uniforms.dst_size.y;
    let idx = chw_index(pos.x, pos.y, 0u, width, height);
    let plane_size = width * height;
    output_buffer[idx] = rgb.b;
    output_buffer[idx + plane_size] = rgb.g;
    output_buffer[idx + plane_size * 2u] = rgb.r;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x >= uniforms.dst_size.x || global_id.y >= uniforms.dst_size.y) {
        return;
    }

    let src_size_f = vec2<f32>(vec2<u32>(uniforms.src_size));
    let drawn_f = vec2<f32>(vec2<u32>(uniforms.drawn));

    // Outside the drawn region there is no source to sample, so write the bars and stop.
    // Scaling x and y independently to avoid this is what experiment 96 removed: it showed
    // the model a face squashed in proportion to how far the source was from the input's
    // aspect, and cost 77 detections in 1239 images.
    let dst_pos = vec2<i32>(global_id.xy) - vec2<i32>(uniforms.origin);
    if (dst_pos.x < 0 || dst_pos.y < 0
        || u32(dst_pos.x) >= uniforms.drawn.x || u32(dst_pos.y) >= uniforms.drawn.y) {
        write_pixel(global_id.xy, vec3<f32>(PAD, PAD, PAD));
        return;
    }

    // Source texels covered by one destination texel. A single bilinear tap only ever blends
    // a 2x2 neighbourhood, so on minification it samples 4 source pixels out of `ratio.x *
    // ratio.y` and the result aliases: a 9x downscale was reading 4 of ~54 pixels, which moved
    // real detections (23px on a landmark) rather than merely losing precision. Average enough
    // taps to cover the box instead.
    //
    // `ceil(ratio)`, not `ceil(ratio * 0.5)`. The halved count was a *coverage* rule -- taps
    // spaced `ratio / n` apart, each spanning ~2 texels, leave no gaps once n >= ratio / 2 --
    // and covering the box is not the same as weighting it like the CPU resize does. It went
    // unnoticed while every non-square source was stretched, because the short axis was then
    // an upscale, where a bilinear tap and a triangle kernel are the same thing. Letterboxing
    // made that axis a downscale too (experiment 96) and the gap showed up as a CPU/GPU
    // detection mismatch: 0.0010 of score and 0.80 px of box, against 0.0004 and 0.23 px with
    // the denser sampling. Only sources under the upload cutoff reach this shader, so the
    // ratios here stay near 2 and the extra taps are a handful of samples.
    let ratio = src_size_f / drawn_f;
    let taps = clamp(
        vec2<u32>(ceil(ratio)),
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
            let src_pos = (vec2<f32>(dst_pos) + offset) * ratio;
            accum = accum + textureSampleLevel(source_tex, source_sampler, src_pos / src_size_f, 0.0);
        }
    }
    let color = accum / (taps_f.x * taps_f.y);
    // Convert from normalized [0,1] floats back to 0-255 range.
    write_pixel(global_id.xy, color.rgb * 255.0);
}
