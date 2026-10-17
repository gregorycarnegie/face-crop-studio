struct HistogramUniforms {
    pixel_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0)
var<storage, read> pixels : array<u32>;

@group(0) @binding(1)
var<storage, read_write> histogram : array<atomic<u32>>;

@group(0) @binding(2)
var<uniform> hist_params : HistogramUniforms;

@compute @workgroup_size(256)
fn build_histogram(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= hist_params.pixel_count) {
        return;
    }

    let value = pixels[index];
    let r_idx = value & 0xFFu;
    let g_idx = (value >> 8u) & 0xFFu;
    let b_idx = (value >> 16u) & 0xFFu;

    atomicAdd(&histogram[r_idx + 0u], 1u);
    atomicAdd(&histogram[g_idx + 256u], 1u);
    atomicAdd(&histogram[b_idx + 512u], 1u);
}

struct CdfUniforms {
    total_pixels : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0)
var<storage, read> histogram_read : array<u32>;

@group(0) @binding(1)
var<storage, read_write> lut_write : array<u32>;

@group(0) @binding(2)
var<uniform> cdf_params : CdfUniforms;

@compute @workgroup_size(1)
fn compute_lut() {
    let total_pixels = cdf_params.total_pixels;
    if (total_pixels == 0u) {
        for (var idx : u32 = 0u; idx < 768u; idx = idx + 1u) {
            lut_write[idx] = 0u;
        }
        return;
    }

    for (var channel : u32 = 0u; channel < 3u; channel = channel + 1u) {
        let base = channel * 256u;
        var cumulative = 0u;
        var cdf_min = 0u;
        var has_min = false;
        for (var i : u32 = 0u; i < 256u; i = i + 1u) {
            cumulative = cumulative + histogram_read[base + i];
            if (!has_min && histogram_read[base + i] > 0u) {
                cdf_min = cumulative;
                has_min = true;
            }
        }

        var cumulative_run = 0u;
        let denominator = max(total_pixels - cdf_min, 1u);
        for (var i : u32 = 0u; i < 256u; i = i + 1u) {
            cumulative_run = cumulative_run + histogram_read[base + i];
            let numerator = cumulative_run - cdf_min;
            let scaled = clamp(f32(max(numerator, 0u)) / f32(denominator), 0.0, 1.0) * 255.0;
            lut_write[base + i] = u32(round(scaled));
        }
    }
}

struct LutUniforms {
    pixel_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0)
var<storage, read_write> pixels_apply : array<u32>;

@group(0) @binding(1)
var<storage, read> lut : array<u32>;

@group(0) @binding(2)
var<uniform> lut_params : LutUniforms;

@compute @workgroup_size(256)
fn apply_equalization(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= lut_params.pixel_count) {
        return;
    }

    let pixel = pixels_apply[index];
    let r_idx = pixel & 0xFFu;
    let g_idx = (pixel >> 8u) & 0xFFu;
    let b_idx = (pixel >> 16u) & 0xFFu;
    let a = (pixel >> 24u) & 0xFFu;

    let r_new = lut[r_idx + 0u] & 0xFFu;
    let g_new = lut[g_idx + 256u] & 0xFFu;
    let b_new = lut[b_idx + 512u] & 0xFFu;

    pixels_apply[index] =
        r_new | (g_new << 8u) | (b_new << 16u) | (a << 24u);
}
