import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Candidate for experiment 32, on top of 41's register rows: four pixels by two rows per
// thread, the two output rows sharing two of their three input rows. Each accumulator keeps
// its own ky order, bit-exact with `depthwise_scalars.wesl`. Coverage 32 16 1.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox_start = global_id.x * 4u;
    let oy = global_id.y * 2u;
    let oc = global_id.z;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    var top = vec4<f32>(bias[oc]);
    var bottom = vec4<f32>(bias[oc]);
    let x = i32(ox_start);
    // Input row r (oy - 1 + r) is kernel row r for the top output row and r - 1 for the bottom.
    for (var r = 0u; r < 4u; r++) {
        let iy = i32(oy) + i32(r) - 1;
        if iy < 0 || iy >= i32(params.input_height) { continue; }
        let base = (oc * params.input_height + u32(iy)) * params.input_width;
        let r0 = load_or_zero(base, x - 1);
        let r1 = load_or_zero(base, x);
        let r2 = load_or_zero(base, x + 1);
        let r3 = load_or_zero(base, x + 2);
        let r4 = load_or_zero(base, x + 3);
        let r5 = load_or_zero(base, x + 4);
        let v0 = vec4<f32>(r0, r1, r2, r3);
        let v1 = vec4<f32>(r1, r2, r3, r4);
        let v2 = vec4<f32>(r2, r3, r4, r5);
        if r < 3u {
            let w = oc * 9u + r * 3u;
            top = fma(v0, vec4<f32>(weights[w]), top);
            top = fma(v1, vec4<f32>(weights[w + 1u]), top);
            top = fma(v2, vec4<f32>(weights[w + 2u]), top);
        }
        if r > 0u {
            let w = oc * 9u + (r - 1u) * 3u;
            bottom = fma(v0, vec4<f32>(weights[w]), bottom);
            bottom = fma(v1, vec4<f32>(weights[w + 1u]), bottom);
            bottom = fma(v2, vec4<f32>(weights[w + 2u]), bottom);
        }
    }
    write_output(ox_start, oy, oc, top);
    if oy + 1u < params.output_height { write_output(ox_start, oy + 1u, oc, bottom); }
}

fn load_or_zero(base: u32, ix: i32) -> f32 {
    return select(0.0, input_tensor[base + u32(ix)], ix >= 0 && ix < i32(params.input_width));
}
