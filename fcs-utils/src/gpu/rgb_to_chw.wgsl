// Turn tightly packed 8-bit RGB into the f32 BGR CHW tensor inference reads.
//
// The companion to preprocess.wgsl, for sources too large to upload whole. That shader
// resizes and converts in one pass from a texture, which needs the *full-resolution* image
// on the GPU -- 40 MB for a 10 MP photo, plus a full-resolution RGBA conversion on the CPU
// to get it there. Past a couple of megapixels that costs more than it saves, so those
// images are resized on the CPU first and only the 640x640 result comes here.
//
// What is left after the resize is a type and layout change: u8 to f32, RGB to BGR,
// interleaved to planar. Doing it here rather than on the CPU uploads 1.2 MB of bytes
// instead of 4.9 MB of floats, and writes straight into the tensor.
//
// No sampling, no filtering, no arithmetic on the values -- each output float is exactly
// the integer value of one source byte, so this is bit-identical to the CPU version rather
// than merely close.

// Same layout as preprocess.wgsl's uniforms so both shaders can share one buffer and one
// host-side struct. Here the source is already at the destination size, so the two match.
struct Dims {
    src_size : vec2<u32>,
    dst_size : vec2<u32>,
};

// Packed as u32 words because WGSL has no u8: byte `i` lives in word `i / 4`, at bit
// `(i % 4) * 8`. Little-endian, matching how the bytes were written on the host.
@group(0) @binding(0)
var<storage, read> source_bytes : array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_buffer : array<f32>;

@group(0) @binding(2)
var<uniform> uniforms : Dims;

fn byte_at(index : u32) -> f32 {
    let word = source_bytes[index >> 2u];
    let shift = (index & 3u) * 8u;
    return f32((word >> shift) & 0xffu);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let plane_size = uniforms.dst_size.x * uniforms.dst_size.y;
    let pixel = global_id.x;
    if (pixel >= plane_size) {
        return;
    }

    // Three interleaved source bytes per pixel, one per plane on the way out.
    let base = pixel * 3u;
    output_buffer[pixel] = byte_at(base + 2u);
    output_buffer[pixel + plane_size] = byte_at(base + 1u);
    output_buffer[pixel + plane_size * 2u] = byte_at(base);
}
