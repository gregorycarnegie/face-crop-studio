struct ShapeMaskUniforms {
    width: u32,
    height: u32,
    point_count: u32,
    samples: u32,
    vignette_softness: f32,
    vignette_intensity: f32,
    vignette_color: u32,
};

@group(0) @binding(0)
var<storage, read_write> pixels : array<u32>;

@group(0) @binding(1)
var<storage, read> polygon_points : array<vec2<f32>>;

@group(0) @binding(2)
var<uniform> params : ShapeMaskUniforms;

fn point_in_polygon(p: vec2<f32>) -> bool {
    var inside = false;
    var i: u32 = 0u;
    var j: u32 = params.point_count - 1u;
    loop {
        if i >= params.point_count {
            break;
        }
        let pi = polygon_points[i];
        let pj = polygon_points[j];
        let intersects = ((pi.y > p.y) != (pj.y > p.y)) && (p.x < fma((pj.x - pi.x), (p.y - pi.y) / (pj.y - pi.y), pi.x));
        inside = select(inside, !inside, intersects);
        j = i;
        i = i + 1u;
    }
    return inside;
}

fn coverage_for_pixel(x: u32, y: u32) -> f32 {
    let samples = max(params.samples, 1u);
    let inv = 1.0 / f32(samples);
    var covered = 0.0;
    for (var i: u32 = 0u; i < samples; i = i + 1u) {
        let offset = fma(vec2<f32>(f32(i & 1u), f32((i >> 1u) & 1u)), vec2<f32>(0.5), vec2<f32>(0.25));
        let sample = vec2<f32>(f32(x), f32(y)) + offset;
        if point_in_polygon(sample) {
            covered = covered + inv;
        }
    }
    return covered;
}

fn sd_polygon(p: vec2<f32>) -> f32 {
    var d = dot(p - polygon_points[0], p - polygon_points[0]);
    var s = 1.0;
    var i: u32 = 0u;
    var j: u32 = params.point_count - 1u;
    loop {
        if i >= params.point_count {
            break;
        }
        let vi = polygon_points[i];
        let vj = polygon_points[j];
        let e = vj - vi;
        let w = p - vi;
        let b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min(d, dot(b, b));

        let c0 = p.y >= vi.y;
        let c1 = p.y < vj.y;
        let c2 = e.x * w.y > e.y * w.x;
        let c = vec3(c0, c1, c2);
        if all(c) || all(!c) {
            s = -s;
        }

        j = i;
        i = i + 1u;
    }
    return s * sqrt(d);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height || params.point_count < 3u {
        return;
    }

    let dst = y * params.width + x;
    var coverage = coverage_for_pixel(x, y);

    if params.vignette_softness > 0.0 {
        let p = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
        let dist = sd_polygon(p);
        // Use a centered fade approach so the vignette encroaches INTO the image
        // radius = total fade width / 2
        let blur_radius = f32(min(params.width, params.height)) * 0.5 * params.vignette_softness;
        let radius = max(blur_radius, 1.0);
        
        // Map dist from [-radius, radius] to coverage [1.0, 0.0]
        // dist / radius gives range [-1, 1] across the fade zone
        let norm_dist = dist / radius;
        let vignette = clamp(0.5 - 0.5 * norm_dist, 0.0, 1.0);

        coverage = coverage * vignette;
    }

    let pixel = pixels[dst];
    let r = f32(pixel & 0xFFu);
    let g = f32((pixel >> 8u) & 0xFFu);
    let b = f32((pixel >> 16u) & 0xFFu);
    let a = f32((pixel >> 24u) & 0xFFu);

    let vr = f32(params.vignette_color & 0xFFu);
    let vg = f32((params.vignette_color >> 8u) & 0xFFu);
    let vb = f32((params.vignette_color >> 16u) & 0xFFu);

    // Vignette color mixing: blend toward vignette color in masked-out regions
    // intensity controls how much vignette color shows, coverage controls where
    let inv_coverage = 1.0 - coverage;
    let r_out = u32(clamp(round(r + inv_coverage * params.vignette_intensity * (vr - r)), 0.0, 255.0)) & 0xFFu;
    let g_out = u32(clamp(round(g + inv_coverage * params.vignette_intensity * (vg - g)), 0.0, 255.0)) & 0xFFu;
    let b_out = u32(clamp(round(b + inv_coverage * params.vignette_intensity * (vb - b)), 0.0, 255.0)) & 0xFFu;

    // Alpha: purely based on coverage, independent of vignette intensity
    // This ensures clean transparent crops regardless of intensity setting
    let a_out = u32(clamp(round(a * coverage), 0.0, 255.0)) & 0xFFu;

    pixels[dst] = r_out | (g_out << 8u) | (b_out << 16u) | (a_out << 24u);
}
