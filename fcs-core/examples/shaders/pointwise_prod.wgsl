// Production's pointwise kernel as a single `main`, for `conv2d_experiment` controls against
// variants (experiment 41). Must stay byte-for-byte the arithmetic of `pointwise` in
// `fcs-core/src/gpu/conv2d.wgsl`.
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
    let ox_start = global_id.x * 4u;
    let oy = global_id.y;
    let oc = global_id.z * 4u;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    pointwise(ox_start, oy, oc);
}

fn pointwise(ox_start: u32, oy: u32, oc: u32) {
    let plane = params.input_width * params.input_height;
    let pixel = oy * params.input_width + ox_start;
    var acc: array<vec4<f32>, 4>;
    for (var j = 0u; j < 4u; j++) {
        if oc + j < params.output_channels { acc[j] = vec4<f32>(bias[oc + j]); }
    }
    for (var ic = 0u; ic < params.input_channels; ic++) {
        let base = ic * plane + pixel;
        var values = vec4<f32>(0.0);
        values.x = input_tensor[base];
        if ox_start + 1u < params.input_width { values.y = input_tensor[base + 1u]; }
        if ox_start + 2u < params.input_width { values.z = input_tensor[base + 2u]; }
        if ox_start + 3u < params.input_width { values.w = input_tensor[base + 3u]; }
        for (var j = 0u; j < 4u; j++) {
            if oc + j < params.output_channels {
                acc[j] = fma(values, vec4<f32>(weights[(oc + j) * params.input_channels + ic]), acc[j]);
            }
        }
    }
    for (var j = 0u; j < 4u; j++) {
        if oc + j < params.output_channels { write_output(ox_start, oy, oc + j, acc[j]); }
    }
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
