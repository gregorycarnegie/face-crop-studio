const MAX_RADIUS : u32 = 8u;

struct BilateralUniforms {
    width : u32,
    height : u32,
    radius : u32,
    sigma_space : f32,
    sigma_color : f32,
    amount : f32,
    _pad0 : f32,
    _pad1 : f32,
}

@group(0) @binding(0)
var<storage, read> input_pixels : array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_pixels : array<u32>;

@group(0) @binding(2)
var<uniform> params : BilateralUniforms;

fn unpack_pixel(value : u32) -> vec4<f32> {
    return vec4<f32>(
        f32(value & 0xFFu),
        f32((value >> 8u) & 0xFFu),
        f32((value >> 16u) & 0xFFu),
        f32((value >> 24u) & 0xFFu),
    );
}

fn pack_pixel(color : vec4<f32>) -> u32 {
    let clamped = clamp(round(color), vec4<f32>(0.0), vec4<f32>(255.0));
    let r = u32(clamped.x) & 0xFFu;
    let g = u32(clamped.y) & 0xFFu;
    let b = u32(clamped.z) & 0xFFu;
    let a = u32(clamped.w) & 0xFFu;
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

fn load_pixel(idx : u32) -> vec4<f32> {
    return unpack_pixel(input_pixels[idx]);
}

fn store_pixel(idx : u32, value : vec4<f32>) {
    output_pixels[idx] = pack_pixel(value);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let dst_index = y * params.width + x;
    let center = load_pixel(dst_index);
    var accum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let sigma_space_term = max(1e-4, 2.0 * params.sigma_space * params.sigma_space);
    let sigma_color_term = max(1e-4, 2.0 * params.sigma_color * params.sigma_color);

    let radius = i32(min(params.radius, MAX_RADIUS));
    var dy = -radius;
    loop {
        if (dy > radius) { break; }
        var dx = -radius;
        loop {
            if (dx > radius) { break; }

            var sample_x = clamp(i32(x) + dx, 0, i32(params.width) - 1);
            var sample_y = clamp(i32(y) + dy, 0, i32(params.height) - 1);
            let sample_idx = (u32(sample_y) * params.width + u32(sample_x));
            let sample = load_pixel(sample_idx);

            let spatial_dist = f32(dx * dx + dy * dy);
            let spatial_weight = exp(-spatial_dist / sigma_space_term);

            let diff = sample.xyz - center.xyz;
            let color_weight = exp(-(dot(diff, diff)) / sigma_color_term);

            let weight = spatial_weight * color_weight;
            accum = fma(sample.xyz, vec3<f32>(weight), accum);
            weight_sum = weight_sum + weight;

            dx = dx + 1;
        }
        dy = dy + 1;
    }
    let filtered = select(center.xyz, accum / weight_sum, weight_sum > 1e-5);
    let amount = clamp(params.amount, 0.0, 1.0);
    let blended = mix(center.xyz, filtered, amount);
    store_pixel(dst_index, vec4<f32>(blended, center.w));
}
