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
