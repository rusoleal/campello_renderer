#pragma once

#include <simd/simd.h>
#include <cmath>
#include <algorithm>

// Arcball orbit camera for the macOS example.
// Coordinate system: right-handed (GLTF convention).
// Projection: Metal NDC (z in [0, 1]).
struct Camera {
    float phi;    // azimuth angle, radians
    float theta;  // elevation angle, radians (clamped away from poles)
    float radius; // distance from target

    simd_float3 target;
    float fovY;   // vertical field of view, radians
    float nearZ;
    float farZ;

    Camera()
        : phi(0.0f)
        , theta(0.35f)
        , radius(3.0f)
        , target(simd_make_float3(0.0f, 0.0f, 0.0f))
        , fovY(M_PI / 4.0f)
        , nearZ(0.01f)
        , farZ(1000.0f)
    {}

    simd_float3 position() const {
        float ct = cosf(theta);
        return simd_make_float3(
            target.x + radius * ct * sinf(phi),
            target.y + radius * sinf(theta),
            target.z + radius * ct * cosf(phi)
        );
    }

    simd_float4x4 viewMatrix() const {
        simd_float3 eye = position();
        simd_float3 f   = simd_normalize(target - eye);
        simd_float3 up  = simd_make_float3(0.0f, 1.0f, 0.0f);
        simd_float3 r   = simd_normalize(simd_cross(f, up));
        simd_float3 u   = simd_cross(r, f);

        return (simd_float4x4){{
            { r.x,  u.x, -f.x, 0.0f },
            { r.y,  u.y, -f.y, 0.0f },
            { r.z,  u.z, -f.z, 0.0f },
            { -simd_dot(r, eye), -simd_dot(u, eye), simd_dot(f, eye), 1.0f }
        }};
    }

    // Metal perspective projection (z in [0, 1], left-handed NDC).
    simd_float4x4 projectionMatrix(float aspect) const {
        float ys = 1.0f / tanf(fovY * 0.5f);
        float xs = ys / aspect;
        float zn = nearZ, zf = farZ;
        float A  = zf / (zn - zf);   // maps z to [0, 1]
        float B  = zn * zf / (zn - zf);

        return (simd_float4x4){{
            { xs,   0.0f,  0.0f, 0.0f },
            { 0.0f, ys,    0.0f, 0.0f },
            { 0.0f, 0.0f,  A,   -1.0f },
            { 0.0f, 0.0f,  B,    0.0f }
        }};
    }

    void orbit(float dPhi, float dTheta) {
        phi   += dPhi;
        theta += dTheta;
        theta = fmaxf(-(float)M_PI / 2.0f + 0.01f,
                fminf( (float)M_PI / 2.0f - 0.01f, theta));
    }

    void zoom(float delta) {
        radius *= (1.0f - delta * 0.1f);
        radius = std::max(0.05f, radius);
    }

    void pan(float dx, float dy) {
        simd_float3 eye = position();
        simd_float3 f   = simd_normalize(target - eye);
        simd_float3 up  = simd_make_float3(0.0f, 1.0f, 0.0f);
        simd_float3 r   = simd_normalize(simd_cross(f, up));
        simd_float3 u   = simd_cross(r, f);
        float scale = radius * 0.001f;
        target = target - r * (dx * scale) + u * (dy * scale);
    }

    // Reset to fit a bounding sphere of the given radius at the origin.
    void fitBounds(float boundsRadius) {
        radius = boundsRadius * 2.5f;
        nearZ  = boundsRadius * 0.01f;
        farZ   = boundsRadius * 100.0f;
    }
};
