import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Candidate for experiment 31: 41's register pointwise kernel reading an NHWC input -- each input
// channel's four values come from four per-pixel runs instead of four adjacent values in a plane.
// Timing only (`FCS_CONV_TIMING_ONLY`): the harness uploads NCHW, and the question is what the
// access pattern costs, which does not depend on the values.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ox_start = id.x * 4u;
    let oy = id.y;
    let oc = id.z * 4u;
    if ox_start >= params.output_width || oy >= params.output_height || oc + 4u > params.output_channels { return; }
    let c_in = params.input_channels;
    let w = params.input_width;
    let p0 = (oy * w + ox_start) * c_in;
    let w0 = oc * c_in;
    let w1 = w0 + c_in;
    let w2 = w1 + c_in;
    let w3 = w2 + c_in;
    var a0 = vec4<f32>(bias[oc]);
    var a1 = vec4<f32>(bias[oc + 1u]);
    var a2 = vec4<f32>(bias[oc + 2u]);
    var a3 = vec4<f32>(bias[oc + 3u]);
    for (var ic = 0u; ic < c_in; ic++) {
        var values = vec4<f32>(0.0);
        values.x = input_tensor[p0 + ic];
        if ox_start + 1u < w { values.y = input_tensor[p0 + c_in + ic]; }
        if ox_start + 2u < w { values.z = input_tensor[p0 + 2u * c_in + ic]; }
        if ox_start + 3u < w { values.w = input_tensor[p0 + 3u * c_in + ic]; }
        a0 = fma(values, vec4<f32>(weights[w0 + ic]), a0);
        a1 = fma(values, vec4<f32>(weights[w1 + ic]), a1);
        a2 = fma(values, vec4<f32>(weights[w2 + ic]), a2);
        a3 = fma(values, vec4<f32>(weights[w3 + ic]), a3);
    }
    write_output(ox_start, oy, oc, a0);
    write_output(ox_start, oy, oc + 1u, a1);
    write_output(ox_start, oy, oc + 2u, a2);
    write_output(ox_start, oy, oc + 3u, a3);
}
