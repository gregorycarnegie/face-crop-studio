// Nearest 2x upsample of `small` added to `skip`, in one dispatch.
//
// The neck's two `resize2x` outputs are read by exactly one consumer each, the `add` that
// follows, so the upsampled tensor never needs to exist. Same single addition per element as
// the separate pair, in the same order (upsampled + skip), so the result is bit-exact.
struct ResizeAddUniforms {
    input_width: u32,
    input_height: u32,
    channels: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> small_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> skip_tensor: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(3) var<uniform> params: ResizeAddUniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_w = params.input_width << 1u;
    let out_h = params.input_height << 1u;
    if id.x >= out_w || id.y >= out_h || id.z >= params.channels {
        return;
    }
    let small = (id.z * params.input_height + (id.y >> 1u)) * params.input_width + (id.x >> 1u);
    let out = (id.z * out_h + id.y) * out_w + id.x;
    output_tensor[out] = small_tensor[small] + skip_tensor[out];
}
