// Candidate for experiment 28: `pointwise_fixed4.wgsl` with the input channel count a
// compile-time constant, so FXC and DXC can see a fixed 64-iteration loop instead of a
// uniform-bounded one. Valid only for 64 -> 64 layers (run with FCS_CONV_CASE); the question is
// the ceiling specialisation could buy, before paying a pipeline per shape to get it.
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

const C_IN: u32 = 64u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox_start = global_id.x * 4u;
    let oy = global_id.y;
    let oc = global_id.z * 4u;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    let plane = params.input_width * params.input_height;
    let pixel = oy * params.input_width + ox_start;
    let w0 = oc * C_IN;
    let w1 = w0 + C_IN;
    let w2 = w1 + C_IN;
    let w3 = w2 + C_IN;
    var a0 = vec4<f32>(bias[oc]);
    var a1 = vec4<f32>(bias[oc + 1u]);
    var a2 = vec4<f32>(bias[oc + 2u]);
    var a3 = vec4<f32>(bias[oc + 3u]);
    for (var ic = 0u; ic < C_IN; ic++) {
        let base = ic * plane + pixel;
        var values = vec4<f32>(0.0);
        values.x = input_tensor[base];
        if ox_start + 1u < params.input_width { values.y = input_tensor[base + 1u]; }
        if ox_start + 2u < params.input_width { values.z = input_tensor[base + 2u]; }
        if ox_start + 3u < params.input_width { values.w = input_tensor[base + 3u]; }
        a0 = fma(values, vec4<f32>(weights[w0 + ic]), a0);
        a1 = fma(values, vec4<f32>(weights[w1 + ic]), a1);
        a2 = fma(values, vec4<f32>(weights[w2 + ic]), a2);
        a3 = fma(values, vec4<f32>(weights[w3 + ic]), a3);
    }
    write_output(ox_start, oy, oc, a0);
    write_output(ox_start, oy, oc + 1u, a1);
    write_output(ox_start, oy, oc + 2u, a2);
    write_output(ox_start, oy, oc + 3u, a3);
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
