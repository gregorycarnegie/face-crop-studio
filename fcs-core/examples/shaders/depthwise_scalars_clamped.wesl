import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Candidate for experiments 41 and 32: production's depthwise with the six-value row held in
// registers instead of `array<f32, 6>`, the pattern 41 found DXC lowering to stack memory.
// Padding comes from `select` over a read clamped into the row, so no index is ever out of
// range and every
// value -- zero or loaded -- is exactly what production computes: bit-exact.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox_start = global_id.x * 4u;
    let oy = global_id.y;
    let oc = global_id.z;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    depthwise(ox_start, oy, oc);
}

fn load_or_zero(base: u32, ix: i32) -> f32 {
    let safe = u32(clamp(ix, 0, i32(params.input_width) - 1));
    return select(0.0, input_tensor[base + safe], ix >= 0 && ix < i32(params.input_width));
}

fn depthwise(ox_start: u32, oy: u32, oc: u32) {
    var acc = vec4<f32>(bias[oc]);
    let x = i32(ox_start);
    for (var ky = 0u; ky < 3u; ky++) {
        let iy = i32(oy) + i32(ky) - 1;
        if iy < 0 || iy >= i32(params.input_height) { continue; }
        let base = (oc * params.input_height + u32(iy)) * params.input_width;
        let r0 = load_or_zero(base, x - 1);
        let r1 = load_or_zero(base, x);
        let r2 = load_or_zero(base, x + 1);
        let r3 = load_or_zero(base, x + 2);
        let r4 = load_or_zero(base, x + 3);
        let r5 = load_or_zero(base, x + 4);
        let w = oc * 9u + ky * 3u;
        acc = fma(vec4<f32>(r0, r1, r2, r3), vec4<f32>(weights[w]), acc);
        acc = fma(vec4<f32>(r1, r2, r3, r4), vec4<f32>(weights[w + 1u]), acc);
        acc = fma(vec4<f32>(r2, r3, r4, r5), vec4<f32>(weights[w + 2u]), acc);
    }
    write_output(ox_start, oy, oc, acc);
}
