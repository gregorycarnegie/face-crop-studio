import package::common::{params, write_output};
import package::bindings_f16::{input_tensor, weights, bias};

// Control for experiment 45: f16 storage with f32 arithmetic, in 41's register form -- the
// candidate experiment 4 rejected, minus the accumulator array 41 found DXC mishandles.
enable f16;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ox_start = id.x * 4u;
    let oy = id.y;
    let oc = id.z * 4u;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels { return; }
    if oc + 4u <= params.output_channels {
        full_tile(ox_start, oy, oc);
    } else {
        for (var j = oc; j < params.output_channels; j++) { one_channel(ox_start, oy, j); }
    }
}

fn load4(ox_start: u32, base: u32) -> vec4<f32> {
    var values = vec4<f32>(0.0);
    values.x = f32(input_tensor[base]);
    if ox_start + 1u < params.input_width { values.y = f32(input_tensor[base + 1u]); }
    if ox_start + 2u < params.input_width { values.z = f32(input_tensor[base + 2u]); }
    if ox_start + 3u < params.input_width { values.w = f32(input_tensor[base + 3u]); }
    return values;
}

fn full_tile(ox_start: u32, oy: u32, oc: u32) {
    let c_in = params.input_channels;
    let plane = params.input_width * params.input_height;
    let pixel = oy * params.input_width + ox_start;
    let w0 = oc * c_in;
    let w1 = w0 + c_in;
    let w2 = w1 + c_in;
    let w3 = w2 + c_in;
    var a0 = vec4<f32>(f32(bias[oc]));
    var a1 = vec4<f32>(f32(bias[oc + 1u]));
    var a2 = vec4<f32>(f32(bias[oc + 2u]));
    var a3 = vec4<f32>(f32(bias[oc + 3u]));
    for (var ic = 0u; ic < c_in; ic++) {
        let values = load4(ox_start, ic * plane + pixel);
        a0 = fma(values, vec4<f32>(f32(weights[w0 + ic])), a0);
        a1 = fma(values, vec4<f32>(f32(weights[w1 + ic])), a1);
        a2 = fma(values, vec4<f32>(f32(weights[w2 + ic])), a2);
        a3 = fma(values, vec4<f32>(f32(weights[w3 + ic])), a3);
    }
    write_output(ox_start, oy, oc, vec4<f32>(a0));
    write_output(ox_start, oy, oc + 1u, vec4<f32>(a1));
    write_output(ox_start, oy, oc + 2u, vec4<f32>(a2));
    write_output(ox_start, oy, oc + 3u, vec4<f32>(a3));
}

// Channel tails are rare in YuNet and only need to be correct.
fn one_channel(ox_start: u32, oy: u32, oc: u32) {
    let c_in = params.input_channels;
    let plane = params.input_width * params.input_height;
    let pixel = oy * params.input_width + ox_start;
    var a = vec4<f32>(f32(bias[oc]));
    for (var ic = 0u; ic < c_in; ic++) {
        a = fma(load4(ox_start, ic * plane + pixel), vec4<f32>(f32(weights[oc * c_in + ic])), a);
    }
    write_output(ox_start, oy, oc, vec4<f32>(a));
}
