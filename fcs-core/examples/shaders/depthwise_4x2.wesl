import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Candidate for experiment 32: a two-dimensional register tile, four pixels by two rows per
// thread. The two output rows share two of their three input rows, so four rows of six loads
// (24) replace two threads' 36. Each accumulator keeps its own ky order, bit-exact.
// Coverage 32 16 1.

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
    // Input row r (oy - 1 + r) is kernel row r for the top output row and r - 1 for the bottom.
    for (var r = 0u; r < 4u; r++) {
        let iy = i32(oy) + i32(r) - 1;
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
            if r < 3u {
                top = fma(values, vec4<f32>(weights[oc * 9u + r * 3u + kx]), top);
            }
            if r > 0u {
                bottom = fma(values, vec4<f32>(weights[oc * 9u + (r - 1u) * 3u + kx]), bottom);
            }
        }
    }
    write_output(ox_start, oy, oc, top);
    if oy + 1u < params.output_height { write_output(ox_start, oy + 1u, oc, bottom); }
}
