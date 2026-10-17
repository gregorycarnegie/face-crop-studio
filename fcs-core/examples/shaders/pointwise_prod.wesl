import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Production's pointwise kernel as a single `main`, for `conv2d_experiment` controls against
// variants (experiment 41). Must stay byte-for-byte the arithmetic of `pointwise` in
// `fcs-core/src/gpu/conv2d.wesl`.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox_start = global_id.x * 4u;
    let oy = global_id.y;
    let oc = global_id.z * 4u;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    pointwise(ox_start, oy, oc);
}

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
