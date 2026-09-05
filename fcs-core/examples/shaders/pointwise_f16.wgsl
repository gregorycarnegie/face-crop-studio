// Experiment only: f16 storage, f32 accumulation, four output channels.
// Binding 0/1/2 must contain packed f16; output remains f32.
enable f16;
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

@group(0) @binding(0) var<storage, read> input_tensor: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2dUniforms;


const ACT_NONE: u32 = 0u;
const ACT_RELU: u32 = 1u;
const ACT_SIGMOID: u32 = 2u;
const ACT_SILU: u32 = 3u;


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ox_start = id.x * 4u;
    let oy = id.y;
    let oc = id.z * 4u;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels { return; }
    let plane = params.input_width * params.input_height;
    let pixel = oy * params.input_width + ox_start;
    var acc: array<vec4<f32>, 4>;
    for (var j = 0u; j < 4u; j++) {
        if oc + j < params.output_channels { acc[j] = vec4<f32>(f32(bias[oc + j])); }
    }
    for (var ic = 0u; ic < params.input_channels; ic++) {
        let base = ic * plane + pixel;
        var values = vec4<f32>(0.0);
        values.x = f32(input_tensor[base]);
        if ox_start + 1u < params.input_width { values.y = f32(input_tensor[base + 1u]); }
        if ox_start + 2u < params.input_width { values.z = f32(input_tensor[base + 2u]); }
        if ox_start + 3u < params.input_width { values.w = f32(input_tensor[base + 3u]); }
        for (var j = 0u; j < 4u; j++) {
            if oc + j < params.output_channels {
                acc[j] = fma(values, vec4<f32>(f32(weights[(oc + j) * params.input_channels + ic])), acc[j]);
            }
        }
    }
    for (var j = 0u; j < 4u; j++) {
        if oc + j < params.output_channels { write_output(ox_start, oy, oc + j, acc[j]); }
    }
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
