#pragma once

namespace systems::leal::gpu
{

    enum class PixelFormat
    {
        invalid = 0,
        // 8-bit formats
        r8unorm = 10,
        r8snorm = 12,
        r8uint = 13,
        r8sint = 14,

        // 16-bit formats
        r16unorm = 20,
        r16snorm = 22,
        r16uint = 23,
        r16sint = 24,
        r16float = 25,
        rg8unorm = 30,
        rg8snorm = 32,
        rg8uint = 33,
        rg8sint = 34,

        // 32-bit formats
        r32uint = 53,
        r32sint = 54,
        r32float = 55,
        rg16unorm = 60,
        rg16snorm = 62,
        rg16uint = 63,
        rg16sint = 64,
        rg16float = 65,
        rgba8unorm = 70,
        rgba8unorm_srgb = 71,
        rgba8snorm = 72,
        rgba8uint = 73,
        rgba8sint = 74,
        bgra8unorm = 80,
        bgra8unorm_srgb = 81,
        // Packed 32-bit formats
        rgb9e5ufloat = 93,
        rgb10a2uint = 91,
        rgb10a2unorm = 90,
        rg11b10ufloat = 92,

        // 64-bit formats
        rg32uint = 103,
        rg32sint = 104,
        rg32float = 105,
        rgba16unorm = 110,
        rgba16snorm = 112,
        rgba16uint = 113,
        rgba16sint = 114,
        rgba16float = 115,

        // 128-bit formats
        rgba32uint = 123,
        rgba32sint = 124,
        rgba32float = 125,

        // Depth/stencil formats
        stencil8 = 253,
        depth16unorm = 250,
        // depth24plus, // no metal compatible
        depth24plus_stencil8 = 255,
        depth32float = 252,

        // "depth32float-stencil8" feature
        depth32float_stencil8 = 260,

        // BC compressed formats usable if "texture-compression-bc" is both
        // supported by the device/user agent and enabled in requestDevice.
        bc1_rgba_unorm = 130,
        bc1_rgba_unorm_srgb = 131,
        bc2_rgba_unorm = 132,
        bc2_rgba_unorm_srgb = 133,
        bc3_rgba_unorm = 134,
        bc3_rgba_unorm_srgb = 135,
        bc4_r_unorm = 140,
        bc4_r_snorm = 141,
        bc5_rg_unorm = 142,
        bc5_rg_snorm = 143,
        bc6h_rgb_ufloat = 151,
        bc6h_rgb_float = 150,
        bc7_rgba_unorm = 152,
        bc7_rgba_unorm_srgb = 153,

        // ETC2 compressed formats usable if "texture-compression-etc2" is both
        // supported by the device/user agent and enabled in requestDevice.
        etc2_rgb8unorm = 180,
        etc2_rgb8unorm_srgb = 181,
        etc2_rgb8a1unorm = 182,
        etc2_rgb8a1unorm_srgb = 183,
        // etc2_rgba8unorm, // no metal compatible type
        // etc2_rgba8unorm_srgb, // no metal compatible type
        eac_r11unorm = 170,
        eac_r11snorm = 172,
        eac_rg11unorm = 174,
        eac_rg11snorm = 176,

        // ASTC compressed formats usable if "texture-compression-astc" is both
        // supported by the device/user agent and enabled in requestDevice.
        // astc_4x4_unorm,
        astc_4x4_unorm_srgb = 186,
        // astc_5x4_unorm,
        astc_5x4_unorm_srgb = 187,
        // astc_5x5_unorm,
        astc_5x5_unorm_srgb = 188,
        // astc_6x5_unorm,
        astc_6x5_unorm_srgb = 189,
        // astc_6x6_unorm,
        astc_6x6_unorm_srgb = 190,
        // astc_8x5_unorm,
        astc_8x5_unorm_srgb = 192,
        // astc_8x6_unorm,
        astc_8x6_unorm_srgb = 193,
        // astc_8x8_unorm,
        astc_8x8_unorm_srgb = 194,
        // astc_10x5_unorm,
        astc_10x5_unorm_srgb = 195,
        // astc_10x6_unorm,
        astc_10x6_unorm_srgb = 196,
        // astc_10x8_unorm,
        astc_10x8_unorm_srgb = 197,
        // astc_10x10_unorm,
        astc_10x10_unorm_srgb = 198,
        // astc_12x10_unorm,
        astc_12x10_unorm_srgb = 199,
        // astc_12x12_unorm,
        astc_12x12_unorm_srgb = 200,
    };

}