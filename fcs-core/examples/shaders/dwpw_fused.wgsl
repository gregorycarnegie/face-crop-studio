// Candidate for experiment 35: depthwise, ReLU and the following pointwise in one dispatch, for a
// depthwise output read by nothing but that pointwise. Per output tile it recomputes the depthwise
// value of every input channel -- 41's register rows -- activates it and accumulates it into four
// pointwise registers, in exactly the order the two separate kernels use, so a real version would
// be bit-exact. Timing only: for a C -> C layer the depthwise weights and biases are read from the
// front of the pointwise buffers, which costs the same as reading their own.
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
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ox_start = id.x * 4u;
    let oy = id.y;
    let oc = id.z * 4u;
    if ox_start >= params.output_width || oy >= params.output_height || oc + 4u > params.output_channels { return; }
    let c = params.input_channels;
    let x = i32(ox_start);
    let w0 = oc * c;
    let w1 = w0 + c;
    let w2 = w1 + c;
    let w3 = w2 + c;
    var a0 = vec4<f32>(bias[oc]);
    var a1 = vec4<f32>(bias[oc + 1u]);
    var a2 = vec4<f32>(bias[oc + 2u]);
    var a3 = vec4<f32>(bias[oc + 3u]);
    for (var ic = 0u; ic < c; ic++) {
        var d = vec4<f32>(bias[ic]);
        for (var ky = 0u; ky < 3u; ky++) {
            let iy = i32(oy) + i32(ky) - 1;
            if iy < 0 || iy >= i32(params.input_height) { continue; }
            let base = (ic * params.input_height + u32(iy)) * params.input_width;
            let r0 = load_or_zero(base, x - 1);
            let r1 = load_or_zero(base, x);
            let r2 = load_or_zero(base, x + 1);
            let r3 = load_or_zero(base, x + 2);
            let r4 = load_or_zero(base, x + 3);
            let r5 = load_or_zero(base, x + 4);
            let wk = ic * 9u + ky * 3u;
            d = fma(vec4<f32>(r0, r1, r2, r3), vec4<f32>(weights[wk]), d);
            d = fma(vec4<f32>(r1, r2, r3, r4), vec4<f32>(weights[wk + 1u]), d);
            d = fma(vec4<f32>(r2, r3, r4, r5), vec4<f32>(weights[wk + 2u]), d);
        }
        d = max(d, vec4<f32>(0.0));
        a0 = fma(d, vec4<f32>(weights[w0 + ic]), a0);
        a1 = fma(d, vec4<f32>(weights[w1 + ic]), a1);
        a2 = fma(d, vec4<f32>(weights[w2 + ic]), a2);
        a3 = fma(d, vec4<f32>(weights[w3 + ic]), a3);
    }
    write_output(ox_start, oy, oc, a0);
    write_output(ox_start, oy, oc + 1u, a1);
    write_output(ox_start, oy, oc + 2u, a2);
    write_output(ox_start, oy, oc + 3u, a3);
}

fn load_or_zero(base: u32, ix: i32) -> f32 {
    let safe = u32(clamp(ix, 0, i32(params.input_width) - 1));
    return select(0.0, input_tensor[base + safe], ix >= 0 && ix < i32(params.input_width));
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
