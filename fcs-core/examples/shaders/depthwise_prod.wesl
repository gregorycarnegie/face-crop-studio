import package::common::{params, write_output};
import package::bindings_f32::{input_tensor, weights, bias};

// Production's depthwise 3x3 kernel as a single `main`, for `conv2d_experiment` controls
// (experiments 32 and 33). Must stay the arithmetic of `depthwise` in
// `fcs-core/src/gpu/conv2d.wesl`.

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
