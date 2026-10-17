struct ActivationUniforms {
    len: u32,
    mode: u32,
};

const MODE_RELU: u32 = 0u;
const MODE_SIGMOID: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> tensor: array<f32>;
@group(0) @binding(1) var<uniform> params: ActivationUniforms;

@compute @workgroup_size(256u, 1u, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.len) {
        return;
    }

    let value = tensor[idx];
    tensor[idx] = select(1.0 / (1.0 + exp(-value)), max(value, 0.0), params.mode == MODE_RELU);
}
