// Experiment only: one 32-lane subgroup reduces channels for four pixels.
// Requires SUBGROUP and a 32-lane subgroup; coverage is 4 x 1 x 1 outputs.
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
    activation_mode: u32, // 0=None, 1=ReLU, 2=Sigmoid, 3=SiLU
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2dUniforms;


const ACT_NONE: u32 = 0u;
const ACT_RELU: u32 = 1u;
const ACT_SIGMOID: u32 = 2u;
const ACT_SILU: u32 = 3u;


@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(subgroup_invocation_id) lane: u32,
        @builtin(subgroup_size) width: u32) {
    let ox_start = group.x * 4u;
    let oy = group.y;
    let oc = group.z;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels { return; }
    let plane = params.input_width * params.input_height;
    let pixel = oy * params.input_width + ox_start;
    var acc = vec4<f32>(0.0);
    for (var ic = lane; ic < params.input_channels; ic += width) {
        let base = ic * plane + pixel;
        var values = vec4<f32>(0.0);
        values.x = input_tensor[base];
        if ox_start + 1u < params.input_width { values.y = input_tensor[base + 1u]; }
        if ox_start + 2u < params.input_width { values.z = input_tensor[base + 2u]; }
        if ox_start + 3u < params.input_width { values.w = input_tensor[base + 3u]; }
        acc = fma(values, vec4<f32>(weights[oc * params.input_channels + ic]), acc);
    }
    let total = subgroupAdd(acc);
    if lane == 0u { write_output(ox_start, oy, oc, total + vec4<f32>(bias[oc])); }
}
fn write_output(ox_start: u32, oy: u32, oc: u32, value: vec4<f32>) {
    var acc = value;
    // Apply fused activation
    if params.activation_mode == ACT_RELU {
        acc = max(acc, vec4<f32>(0.0));
    } else if params.activation_mode == ACT_SIGMOID {
        acc = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-acc));
    } else if params.activation_mode == ACT_SILU {
        acc = acc / (vec4<f32>(1.0) + exp(-acc));
    }

    // Write output
    let out_row_base = (oc * params.output_height + oy) * params.output_width;

    // Pixel 0
    if ox_start < params.output_width {
        output_tensor[out_row_base + ox_start] = acc.x;
    }
    // Pixel 1
    if ox_start + 1u < params.output_width {
        output_tensor[out_row_base + ox_start + 1u] = acc.y;
    }
    // Pixel 2
    if ox_start + 2u < params.output_width {
        output_tensor[out_row_base + ox_start + 2u] = acc.z;
    }
    // Pixel 3
    if ox_start + 3u < params.output_width {
        output_tensor[out_row_base + ox_start + 3u] = acc.w;
    }
}
