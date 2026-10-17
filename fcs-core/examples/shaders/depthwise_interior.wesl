import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Candidate for experiment 33: production's depthwise kernel, plus an interior fast path for
// tiles whose whole 3x3 window lies inside the map -- six loads per row with no bounds tests,
// no skipped rows and no zero padding. Same `fma` sequence (bias, then ky, kx), so bit-exact
// with `depthwise_prod.wesl`. The branch is taken inside the kernel rather than as a separate
// border dispatch, because 8 measured a dispatch at 1.85 us before it does any work.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox_start = global_id.x * 4u;
    let oy = global_id.y;
    let oc = global_id.z;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    let interior = ox_start >= 1u && ox_start + 5u <= params.input_width
        && oy >= 1u && oy + 2u <= params.input_height;
    if interior {
        depthwise_interior(ox_start, oy, oc);
    } else {
        depthwise(ox_start, oy, oc);
    }
}

fn depthwise_interior(ox_start: u32, oy: u32, oc: u32) {
    var acc = vec4<f32>(bias[oc]);
    let w = oc * 9u;
    for (var ky = 0u; ky < 3u; ky++) {
        let base = (oc * params.input_height + oy + ky - 1u) * params.input_width + ox_start - 1u;
        let r0 = input_tensor[base];
        let r1 = input_tensor[base + 1u];
        let r2 = input_tensor[base + 2u];
        let r3 = input_tensor[base + 3u];
        let r4 = input_tensor[base + 4u];
        let r5 = input_tensor[base + 5u];
        let wk = w + ky * 3u;
        acc = fma(vec4<f32>(r0, r1, r2, r3), vec4<f32>(weights[wk]), acc);
        acc = fma(vec4<f32>(r1, r2, r3, r4), vec4<f32>(weights[wk + 1u]), acc);
        acc = fma(vec4<f32>(r2, r3, r4, r5), vec4<f32>(weights[wk + 2u]), acc);
    }
    write_output(ox_start, oy, oc, acc);
}

fn depthwise(ox_start: u32, oy: u32, oc: u32) {
    var acc = vec4<f32>(bias[oc]);
    for (var ky = 0u; ky < 3u; ky++) {
        let iy = i32(oy) + i32(ky) - 1;
        if iy < 0 || iy >= i32(params.input_height) { continue; }
        let base = (oc * params.input_height + u32(iy)) * params.input_width;
        var row: array<f32, 6>;
        for (var i = 0u; i < 6u; i++) {
            let ix = i32(ox_start) + i32(i) - 1;
            row[i] = 0.0;
            if ix >= 0 && ix < i32(params.input_width) { row[i] = input_tensor[base + u32(ix)]; }
        }
        for (var kx = 0u; kx < 3u; kx++) {
            let values = vec4<f32>(row[kx], row[kx + 1u], row[kx + 2u], row[kx + 3u]);
            acc = fma(values, vec4<f32>(weights[oc * 9u + ky * 3u + kx]), acc);
        }
    }
    write_output(ox_start, oy, oc, acc);
}
