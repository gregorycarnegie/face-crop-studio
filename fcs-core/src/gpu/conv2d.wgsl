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

const WORKGROUP_SIZE_X: u32 = 8u;
const WORKGROUP_SIZE_Y: u32 = 8u;

const ACT_NONE: u32 = 0u;
const ACT_RELU: u32 = 1u;
const ACT_SIGMOID: u32 = 2u;
const ACT_SILU: u32 = 3u;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread computes 4 output pixels horizontally
    let ox_start = global_id.x * 4u;
    let oy = global_id.y;
    let pointwise_mode = params.kernel_width == 1u && params.kernel_height == 1u &&
       params.stride_x == 1u && params.stride_y == 1u &&
       params.pad_x == 0u && params.pad_y == 0u && params.groups == 1u;
    let oc = global_id.z * select(1u, 4u, pointwise_mode);

    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }

    if pointwise_mode {
        pointwise(ox_start, oy, oc);
        return;
    }

    if params.kernel_width == 3u && params.kernel_height == 3u &&
       params.stride_x == 1u && params.stride_y == 1u &&
       params.pad_x == 1u && params.pad_y == 1u &&
       params.groups == params.input_channels && params.output_channels == params.input_channels {
        depthwise(ox_start, oy, oc);
        return;
    }

    let group_out = params.output_channels / params.groups;
    let group_in = params.input_channels / params.groups;
    let group_idx = oc / group_out;
    let input_channel_base = group_idx * group_in;

    let kernel_hw = params.kernel_height * params.kernel_width;
    let weights_per_out = group_in * kernel_hw;
    let input_plane = params.input_width * params.input_height;
    let stride = vec2<i32>(i32(params.stride_x), i32(params.stride_y));
    let pad = vec2<i32>(i32(params.pad_x), i32(params.pad_y));
    let dims = vec2<i32>(i32(params.input_width), i32(params.input_height));
    
    // Calculate start positions for 4 pixels
    // ox_vec = [ox, ox+1, ox+2, ox+3]
    let ox_vec = vec4<i32>(i32(ox_start), i32(ox_start) + 1, i32(ox_start) + 2, i32(ox_start) + 3);
    let start_y = i32(oy) * stride.y - pad.y;
    let start_x_vec = ox_vec * stride.x - pad.x;

    // Initialize accumulator with bias
    var acc = vec4<f32>(bias[oc]);

    var local_ic: u32 = 0u;
    loop {
        if local_ic >= group_in {
            break;
        }
        let channel = input_channel_base + local_ic;
        let channel_base = channel * input_plane;
        let weight_channel_base = oc * weights_per_out + local_ic * kernel_hw;

        var ky: u32 = 0u;
        loop {
            if ky >= params.kernel_height {
                break;
            }
            let iy = start_y + i32(ky);
            if iy < 0 || iy >= dims.y {
                ky = ky + 1u;
                continue;
            }
            let input_row_base = channel_base + u32(iy) * params.input_width;
            let weight_row_base = weight_channel_base + ky * params.kernel_width;

            var kx: u32 = 0u;
            loop {
                if kx >= params.kernel_width {
                    break;
                }
                
                // Load weight (scalar broadcast)
                let weight_index = weight_row_base + kx;
                let w = weights[weight_index];
                let w_vec = vec4<f32>(w);

                // Calculate input X coordinates for 4 pixels
                let ix_vec = start_x_vec + i32(kx);
                
                // Gather inputs
                var inputs = vec4<f32>(0.0);
                
                // Unroll manually for 4 components
                // Pixel 0
                if ix_vec.x >= 0 && ix_vec.x < dims.x {
                    inputs.x = input_tensor[input_row_base + u32(ix_vec.x)];
                }
                // Pixel 1
                if ix_vec.y >= 0 && ix_vec.y < dims.x {
                    inputs.y = input_tensor[input_row_base + u32(ix_vec.y)];
                }
                // Pixel 2
                if ix_vec.z >= 0 && ix_vec.z < dims.x {
                    inputs.z = input_tensor[input_row_base + u32(ix_vec.z)];
                }
                // Pixel 3
                if ix_vec.w >= 0 && ix_vec.w < dims.x {
                    inputs.w = input_tensor[input_row_base + u32(ix_vec.w)];
                }

                acc = fma(inputs, w_vec, acc);

                kx = kx + 1u;
            }
            ky = ky + 1u;
        }

        local_ic = local_ic + 1u;
    }

    write_output(ox_start, oy, oc, acc);
}

// Four output channels share each loaded input vector. Dispatch z covers ceil(channels / 4).
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

fn depthwise(ox_start: u32, oy: u32, oc: u32) {
    var acc = vec4<f32>(bias[oc]);
    for (var ky = 0u; ky < 3u; ky++) {
        let iy = i32(oy) + i32(ky) - 1;
        if iy < 0 || iy >= i32(params.input_height) { continue; }
        let base = (oc * params.input_height + u32(iy)) * params.input_width;
        var row: array<f32, 6>;
        for (var i = 0u; i < 6u; i++) {
            let ix = i32(ox_start) + i32(i) - 1;
            row[i] = 0.0;
            if ix >= 0 && ix < i32(params.input_width) { row[i] = input_tensor[base + u32(ix)]; }
        }
        for (var kx = 0u; kx < 3u; kx++) {
            let values = vec4<f32>(row[kx], row[kx + 1u], row[kx + 2u], row[kx + 3u]);
            acc = fma(values, vec4<f32>(weights[oc * 9u + ky * 3u + kx]), acc);
        }
    }
    write_output(ox_start, oy, oc, acc);
}
