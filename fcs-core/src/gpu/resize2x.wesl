struct UpsampleUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
}

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(2) var<uniform> params: UpsampleUniforms;

const RESIZE_WG_X: u32 = 8u;
const RESIZE_WG_Y: u32 = 8u;

fn flatten_index(width: u32, height: u32, channel: u32, y: u32, x: u32) -> u32 {
    return (channel * height + y) * width + x;
}

@compute @workgroup_size(RESIZE_WG_X, RESIZE_WG_Y, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox = global_id.x;
    let oy = global_id.y;
    let oc = global_id.z;

    let output_width = params.input_width << 1u;
    let output_height = params.input_height << 1u;

    if ox >= output_width || oy >= output_height || oc >= params.channels {
        return;
    }

    let ix = ox >> 1u;
    let iy = oy >> 1u;

    let in_index = flatten_index(params.input_width, params.input_height, oc, iy, ix);
    let out_index = flatten_index(output_width, output_height, oc, oy, ox);
    output_tensor[out_index] = input_tensor[in_index];
}
