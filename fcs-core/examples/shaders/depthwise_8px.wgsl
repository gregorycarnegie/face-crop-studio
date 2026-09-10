// Candidate for experiment 32: eight output pixels per thread instead of four. One row of ten
// loads feeds two vec4 accumulators (pixels 0-3 and 4-7), so each input value serves up to six
// multiply-adds instead of three. Same `fma` sequence per pixel, bit-exact. Coverage 64 8 1.
struct Conv2dUniforms {
    input_width: u32,
    input_height: u32,
    input_channels: u32,
    output_width: u32,
    output_height: u32,
    output_channels: u32,
    kernel_width: u32,
    kernel_height: u32,
    stride_x: u32,
    stride_y: u32,
    pad_x: u32,
    pad_y: u32,
    groups: u32,
    activation_mode: u32,
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2dUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox_start = global_id.x * 8u;
    let oy = global_id.y;
    let oc = global_id.z;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    var a = vec4<f32>(bias[oc]);
    var b = vec4<f32>(bias[oc]);
    for (var ky = 0u; ky < 3u; ky++) {
        let iy = i32(oy) + i32(ky) - 1;
        if iy < 0 || iy >= i32(params.input_height) { continue; }
        let base = (oc * params.input_height + u32(iy)) * params.input_width;
        var row: array<f32, 10>;
        for (var i = 0u; i < 10u; i++) {
            let ix = i32(ox_start) + i32(i) - 1;
            row[i] = 0.0;
            if ix >= 0 && ix < i32(params.input_width) { row[i] = input_tensor[base + u32(ix)]; }
        }
        for (var kx = 0u; kx < 3u; kx++) {
            let w = vec4<f32>(weights[oc * 9u + ky * 3u + kx]);
            a = fma(vec4<f32>(row[kx], row[kx + 1u], row[kx + 2u], row[kx + 3u]), w, a);
            b = fma(vec4<f32>(row[kx + 4u], row[kx + 5u], row[kx + 6u], row[kx + 7u]), w, b);
        }
    }
    write_output(ox_start, oy, oc, a);
    write_output(ox_start + 4u, oy, oc, b);
}

fn write_output(ox_start: u32, oy: u32, oc: u32, value: vec4<f32>) {
    var acc = value;
    if params.activation_mode == 1u {
        acc = max(acc, vec4<f32>(0.0));
    } else if params.activation_mode == 2u {
        acc = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-acc));
    } else if params.activation_mode == 3u {
        acc = acc / (vec4<f32>(1.0) + exp(-acc));
    }
    let out_row_base = (oc * params.output_height + oy) * params.output_width;
    if ox_start < params.output_width { output_tensor[out_row_base + ox_start] = acc.x; }
    if ox_start + 1u < params.output_width { output_tensor[out_row_base + ox_start + 1u] = acc.y; }
    if ox_start + 2u < params.output_width { output_tensor[out_row_base + ox_start + 2u] = acc.z; }
    if ox_start + 3u < params.output_width { output_tensor[out_row_base + ox_start + 3u] = acc.w; }
}
