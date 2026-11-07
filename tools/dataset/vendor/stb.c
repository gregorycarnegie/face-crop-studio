// The stb implementations the dataset tools link: `zig run -lc tool.zig tools/dataset/vendor/stb.c`.
// stb_image.h and stb_image_write.h are public domain, from github.com/nothings/stb at
// 013ac3beddff3dbffafd5177e7972067cd2b5083 (stb_image v2.30, stb_image_write v1.16).

// Zig builds C with UBSan in Debug; stb's JPEG writer left-shifts signed ints (benign, upstream).
#pragma clang attribute push(__attribute__((no_sanitize("undefined"))), apply_to = function)

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_WRITE_NO_STDIO
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#pragma clang attribute pop
