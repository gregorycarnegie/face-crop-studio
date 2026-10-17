import package::pixel::{unpack_pixel, pack_pixel};

const MAX_KERNEL_SIZE : u32 = 25u;

struct BlurUniforms {
    width : u32,
    height : u32,
    radius : u32,
    direction : u32,
};

@group(0) @binding(0)
var<storage, read> input_pixels : array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_pixels : array<u32>;

@group(0) @binding(2)
var<uniform> params : BlurUniforms;

@group(0) @binding(3)
var<storage, read> weights : array<f32>;

fn load_pixel(idx : u32) -> vec4<f32> {
    return unpack_pixel(input_pixels[idx]);
}

fn store_pixel(idx : u32, value : vec4<f32>) {
    output_pixels[idx] = pack_pixel(value);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let dst_index = y * params.width + x;
    let center = load_pixel(dst_index);
    var accum = vec3<f32>(0.0);
    let alpha = center.w;

    let radius = i32(params.radius);
    let dims = vec2<i32>(i32(params.width), i32(params.height));
    let max_index = dims - vec2<i32>(1);
    let base = vec2<i32>(i32(x), i32(y));
    let dir_mask = i32(params.direction);
    let dir = vec2<i32>(1 - dir_mask, dir_mask);

    for (var offset : i32 = -radius; offset <= radius; offset = offset + 1) {
        let kernel_index = u32(offset + radius);
        let weight = weights[kernel_index];

        let offset_vec = vec2<i32>(offset, offset) * dir;
        let sample_pos = clamp(base + offset_vec, vec2<i32>(0), max_index);
        let sample_idx = u32(sample_pos.y) * params.width + u32(sample_pos.x);
        let sample = load_pixel(sample_idx);
        accum = fma(sample.xyz, vec3<f32>(weight), accum);
    }

    store_pixel(dst_index, vec4<f32>(accum, alpha));
}
