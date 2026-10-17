import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Candidate for experiment 31: 41's register-row depthwise kernel reading an NHWC input, where a
// row's six neighbours are one channel-width apart instead of adjacent. Timing only.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let ox_start = id.x * 4u;
    let oy = id.y;
    let oc = id.z;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels { return; }
    var acc = vec4<f32>(bias[oc]);
    let x = i32(ox_start);
    for (var ky = 0u; ky < 3u; ky++) {
        let iy = i32(oy) + i32(ky) - 1;
        if iy < 0 || iy >= i32(params.input_height) { continue; }
        let row = u32(iy) * params.input_width;
        let r0 = load_or_zero(row, x - 1, oc);
        let r1 = load_or_zero(row, x, oc);
        let r2 = load_or_zero(row, x + 1, oc);
        let r3 = load_or_zero(row, x + 2, oc);
        let r4 = load_or_zero(row, x + 3, oc);
        let r5 = load_or_zero(row, x + 4, oc);
        let wk = oc * 9u + ky * 3u;
        acc = fma(vec4<f32>(r0, r1, r2, r3), vec4<f32>(weights[wk]), acc);
        acc = fma(vec4<f32>(r1, r2, r3, r4), vec4<f32>(weights[wk + 1u]), acc);
        acc = fma(vec4<f32>(r2, r3, r4, r5), vec4<f32>(weights[wk + 2u]), acc);
    }
    write_output(ox_start, oy, oc, acc);
}

fn load_or_zero(row: u32, ix: i32, oc: u32) -> f32 {
    let safe = u32(clamp(ix, 0, i32(params.input_width) - 1));
    return select(0.0, input_tensor[(row + safe) * params.input_channels + oc],
                  ix >= 0 && ix < i32(params.input_width));
}
