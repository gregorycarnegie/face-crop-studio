import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Candidate for experiment 32: eight output pixels per thread instead of four. One row of ten
// loads feeds two vec4 accumulators (pixels 0-3 and 4-7), so each input value serves up to six
// multiply-adds instead of three. Same `fma` sequence per pixel, bit-exact. Coverage 64 8 1.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ox_start = global_id.x * 8u;
    let oy = global_id.y;
    let oc = global_id.z;
    if ox_start >= params.output_width || oy >= params.output_height || oc >= params.output_channels {
        return;
    }
    var a = vec4<f32>(bias[oc]);
    var b = vec4<f32>(bias[oc]);
    for (var ky = 0u; ky < 3u; ky++) {
        let iy = i32(oy) + i32(ky) - 1;
        if iy < 0 || iy >= i32(params.input_height) { continue; }
        let base = (oc * params.input_height + u32(iy)) * params.input_width;
        var row: array<f32, 10>;
        for (var i = 0u; i < 10u; i++) {
            let ix = i32(ox_start) + i32(i) - 1;
            row[i] = 0.0;
            if ix >= 0 && ix < i32(params.input_width) { row[i] = input_tensor[base + u32(ix)]; }
        }
        for (var kx = 0u; kx < 3u; kx++) {
            let w = vec4<f32>(weights[oc * 9u + ky * 3u + kx]);
            a = fma(vec4<f32>(row[kx], row[kx + 1u], row[kx + 2u], row[kx + 3u]), w, a);
            b = fma(vec4<f32>(row[kx + 4u], row[kx + 5u], row[kx + 6u], row[kx + 7u]), w, b);
        }
    }
    write_output(ox_start, oy, oc, a);
    write_output(ox_start + 4u, oy, oc, b);
}
