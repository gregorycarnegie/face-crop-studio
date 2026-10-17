struct PixelAdjustUniforms {
    exposure_multiplier : f32,
    brightness_offset : f32,
    contrast_factor : f32,
    saturation : f32,
    pixel_count : u32,
    flags : u32,
    _pad0 : u32,
    _pad1 : u32,
};

@group(0) @binding(0)
var<storage, read_write> pixels : array<u32>;

@group(0) @binding(1)
var<uniform> params : PixelAdjustUniforms;

const FLAG_EXPOSURE : u32 = 1u << 0u;
const FLAG_BRIGHTNESS : u32 = 1u << 1u;
const FLAG_CONTRAST : u32 = 1u << 2u;
const FLAG_SATURATION : u32 = 1u << 3u;
const INV_255 : f32 = 1.0 / 255.0;

fn has_flag(bits : u32, flag : u32) -> bool {
    return (bits & flag) != 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.pixel_count) {
        return;
    }

    let pixel = pixels[index];
    var r = f32(pixel & 0xFFu);
    var g = f32((pixel >> 8u) & 0xFFu);
    var b = f32((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;

    var rgb = vec3<f32>(r, g, b);
    let flags = params.flags;

    if (has_flag(flags, FLAG_EXPOSURE)) {
        rgb *= params.exposure_multiplier;
    }

    if (has_flag(flags, FLAG_BRIGHTNESS)) {
        rgb += vec3<f32>(params.brightness_offset);
    }

    if (has_flag(flags, FLAG_CONTRAST)) {
        let normalized = rgb * INV_255 - vec3<f32>(0.5);
        rgb = (normalized * params.contrast_factor + vec3<f32>(0.5)) * 255.0;
    }

    if (has_flag(flags, FLAG_SATURATION)) {
        let gray = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
        rgb = mix(vec3<f32>(gray), rgb, vec3<f32>(params.saturation));
    }

    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(255.0));
    let rounded = round(rgb);

    let r_u32 = u32(rounded.x) & 0xFFu;
    let g_u32 = u32(rounded.y) & 0xFFu;
    let b_u32 = u32(rounded.z) & 0xFFu;
    pixels[index] = r_u32 | (g_u32 << 8u) | (b_u32 << 16u) | (a << 24u);
}
