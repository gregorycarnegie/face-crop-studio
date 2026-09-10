// Candidate for experiments 41 and 32: production's depthwise with the six-value row held in
// registers instead of `array<f32, 6>`, the pattern 41 found DXC lowering to stack memory.
// Padding comes from `select` over a read clamped into the row, so no index is ever out of
// range and every
// value -- zero or loaded -- is exactly what production computes: bit-exact.
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
    let oc = global_id.z;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    depthwise(ox_start, oy, oc);
}

fn load_or_zero(base: u32, ix: i32) -> f32 {
    let safe = u32(clamp(ix, 0, i32(params.input_width) - 1));
    return select(0.0, input_tensor[base + safe], ix >= 0 && ix < i32(params.input_width));
}

fn depthwise(ox_start: u32, oy: u32, oc: u32) {
    var acc = vec4<f32>(bias[oc]);
    let x = i32(ox_start);
    for (var ky = 0u; ky < 3u; ky++) {
        let iy = i32(oy) + i32(ky) - 1;
        if iy < 0 || iy >= i32(params.input_height) { continue; }
        let base = (oc * params.input_height + u32(iy)) * params.input_width;
        let r0 = load_or_zero(base, x - 1);
        let r1 = load_or_zero(base, x);
        let r2 = load_or_zero(base, x + 1);
        let r3 = load_or_zero(base, x + 2);
        let r4 = load_or_zero(base, x + 3);
        let r5 = load_or_zero(base, x + 4);
        let w = oc * 9u + ky * 3u;
        acc = fma(vec4<f32>(r0, r1, r2, r3), vec4<f32>(weights[w]), acc);
        acc = fma(vec4<f32>(r1, r2, r3, r4), vec4<f32>(weights[w + 1u]), acc);
        acc = fma(vec4<f32>(r2, r3, r4, r5), vec4<f32>(weights[w + 2u]), acc);
    }
    write_output(ox_start, oy, oc, acc);
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
