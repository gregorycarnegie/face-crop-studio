import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Experiment only: one 32-lane subgroup reduces channels for four pixels.
// Requires SUBGROUP and a 32-lane subgroup; coverage is 4 x 1 x 1 outputs.

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
