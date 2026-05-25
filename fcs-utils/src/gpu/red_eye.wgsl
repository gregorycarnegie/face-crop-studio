struct RedEyeUniforms {
    pixel_count : u32,
    width : u32,
    threshold : f32,
    min_red : f32,
    eye_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

struct RedEye {
    x : f32,
    y : f32,
    radius : f32,
    _pad : f32,
};

@group(0) @binding(0)
var<storage, read_write> pixels : array<u32>;

@group(0) @binding(1)
var<uniform> params : RedEyeUniforms;

@group(0) @binding(2)
var<storage, read> eyes : array<RedEye>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.pixel_count) {
        return;
    }

    // Check if pixel is within any eye region
    if (params.eye_count > 0u) {
        let y = f32(index / params.width);
        let x = f32(index % params.width);
        
        var in_eye = false;
        for (var i = 0u; i < params.eye_count; i = i + 1u) {
            let eye = eyes[i];
            let dx = x - eye.x;
            let dy = y - eye.y;
            if (dx * dx + dy * dy <= eye.radius * eye.radius) {
                in_eye = true;
                break;
            }
        }
        
        if (!in_eye) {
            return;
        }
    }

    let pixel = pixels[index];
    let r = f32(pixel & 0xFFu);
    // Optimization: if red is low, we don't need to unpack the rest
    if (r <= params.min_red) {
        return;
    }

    let g = f32((pixel >> 8u) & 0xFFu);
    let b = f32((pixel >> 16u) & 0xFFu);
    let a = (pixel >> 24u) & 0xFFu;

    let avg_gb = fma((g + b), 0.5, 1e-6);

    // We already checked r > min_red above (r > 80.0 usually)
    let out_r = select(r, avg_gb, r > avg_gb * params.threshold);
    
    // Only write back if changed
    if (out_r != r) {
        let r_new = u32(clamp(round(out_r), 0.0, 255.0)) & 0xFFu;
        // g, b, a are unchanged
        let g_new = u32(g);
        let b_new = u32(b);
        pixels[index] = r_new | (g_new << 8u) | (b_new << 16u) | (a << 24u);
    }
}
