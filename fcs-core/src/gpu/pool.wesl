struct PoolUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
    output_width: u32,
    output_height: u32,
    kernel: u32,
    stride: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(2) var<uniform> params: PoolUniforms;

const WG_SIZE_X: u32 = 8u;
const WG_SIZE_Y: u32 = 8u;

@compute @workgroup_size(WG_SIZE_X, WG_SIZE_Y, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox = global_id.x;
    let oy = global_id.y;
    let oc = global_id.z;

    if (ox >= params.output_width || oy >= params.output_height || oc >= params.channels) {
        return;
    }

    let stride = i32(params.stride);
    let pad = i32(params.pad);
    let dims = vec2<i32>(i32(params.input_width), i32(params.input_height));
    let start_xy = vec2<i32>(i32(ox), i32(oy)) * vec2<i32>(stride, stride) - vec2<i32>(pad, pad);
    let channel_base = oc * params.input_width * params.input_height;

    var max_val = -3.402823e+38;
    var ky: u32 = 0u;
    loop {
        if (ky >= params.kernel) {
            break;
        }
        let iy = start_xy.y + i32(ky);
        if (iy < 0 || iy >= dims.y) {
            ky = ky + 1u;
            continue;
        }
        let row_offset = channel_base + u32(iy) * params.input_width;

        var kx: u32 = 0u;
        loop {
            if (kx >= params.kernel) {
                break;
            }
            let ix = start_xy.x + i32(kx);
            if (ix >= 0 && ix < dims.x) {
                let value = input_tensor[row_offset + u32(ix)];
                max_val = max(value, max_val);
            }
            kx = kx + 1u;
        }
        ky = ky + 1u;
    }

    let channel_offset = oc * params.output_height + oy;
    let out_index = channel_offset * params.output_width + ox;
    output_tensor[out_index] = max_val;
}
