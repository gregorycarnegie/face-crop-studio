struct AddUniforms {
    len: u32,
}

@group(0) @binding(0) var<storage, read> lhs_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> rhs_tensor: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(3) var<uniform> params: AddUniforms;

const ADD_WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(ADD_WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.len) {
        return;
    }
    output_tensor[idx] = lhs_tensor[idx] + rhs_tensor[idx];
}
