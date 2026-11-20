# Dataset image codecs

`stb.zig` is a Zig source port of the JPEG, PNG and BMP decoders and JPEG/PNG writers
previously built from stb_image 2.30 and stb_image_write 1.16. It started from `zig translate-c`
with `STBI_NO_SIMD`, with unused APIs/platform declarations removed, numeric tables compacted,
C comma-expression control flow corrected, and memory operations moved to Zig allocators.
PNG DEFLATE uses the Zig standard library instead of stb's stretchy-buffer compressor.
The low-level codec names and arithmetic remain recognizable for comparison with upstream.

Upstream: <https://github.com/nothings/stb/tree/013ac3beddff3dbffafd5177e7972067cd2b5083>.
The upstream MIT/public-domain licence is in `LICENSE`.

Use `../image.zig` for the allocator-owning RGB API and EXIF orientation. It scopes the codec's
thread-local allocator to each call, including cleanup; do not keep codec-owned pointers
across allocator changes. The port compiles without C headers, a C compiler or libc.

Run `zig test tools/dataset/image.zig` from the repository root. PNG output is lossless but
compressed bytes differ from the old writer; JPEG output may differ slightly due to floating
point contraction. This scalar port has no hand-written SIMD kernels.
