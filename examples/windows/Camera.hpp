#pragma once

#include <algorithm>
#include <cmath>

// Arcball orbit camera for the Windows example — identical to examples/linux/Camera.hpp
// (pure math, no platform dependency), copied rather than shared to keep each example
// self-contained like the other platform directories.
// Coordinate system: right-handed (GLTF convention).
// Projection: Vulkan/Metal-shared NDC (z in [0, 1]) — see Renderer::setCameraMatrices()'s
// doc comment: it expects column-major float[16] matrices in that same convention.
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

inline float dot(const Vec3 &a, const Vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline Vec3 normalize(const Vec3 &v) {
    float len = std::sqrt(dot(v, v));
    if (len < 1e-8f) return v;
    return v * (1.0f / len);
}

struct Camera {
    float phi;    // azimuth angle, radians
    float theta;  // elevation angle, radians (clamped away from poles)
    float radius; // distance from target

    Vec3  target;
    float fovY;   // vertical field of view, radians
    float nearZ;
    float farZ;

    Camera()
        : phi(0.0f)
        , theta(0.35f)
        , radius(3.0f)
        , target{0.0f, 0.0f, 0.0f}
        , fovY(float(M_PI) / 4.0f)
        , nearZ(0.01f)
        , farZ(1000.0f)
    {}

    Vec3 position() const {
        float ct = std::cos(theta);
        return {
            target.x + radius * ct * std::sin(phi),
            target.y + radius * std::sin(theta),
            target.z + radius * ct * std::cos(phi)
        };
    }

    // Writes a column-major float[16] view matrix into `out`.
    void viewMatrix(float out[16]) const {
        Vec3 eye = position();
        Vec3 f   = normalize(target - eye);
        Vec3 up{0.0f, 1.0f, 0.0f};
        Vec3 r   = normalize(cross(f, up));
        Vec3 u   = cross(r, f);

        // Column-major: out[col*4 + row].
        out[0] = r.x;  out[1] = u.x;  out[2] = -f.x; out[3] = 0.0f;
        out[4] = r.y;  out[5] = u.y;  out[6] = -f.y; out[7] = 0.0f;
        out[8] = r.z;  out[9] = u.z;  out[10] = -f.z; out[11] = 0.0f;
        out[12] = -dot(r, eye);
        out[13] = -dot(u, eye);
        out[14] = dot(f, eye);
        out[15] = 1.0f;
    }

    // Writes a column-major float[16] perspective projection into `out`
    // (z in [0, 1], left-handed NDC — matches Metal's and Vulkan's shared convention).
    void projectionMatrix(float aspect, float out[16]) const {
        float ys = 1.0f / std::tan(fovY * 0.5f);
        float xs = ys / aspect;
        float zn = nearZ, zf = farZ;
        float A  = zf / (zn - zf);   // maps z to [0, 1]
        float B  = zn * zf / (zn - zf);

        for (int i = 0; i < 16; i++) out[i] = 0.0f;
        out[0]  = xs;
        out[5]  = ys;
        out[10] = A;
        out[11] = -1.0f;
        out[14] = B;
    }

    void orbit(float dPhi, float dTheta) {
        phi   += dPhi;
        theta += dTheta;
        theta = std::max(-float(M_PI) / 2.0f + 0.01f,
                 std::min( float(M_PI) / 2.0f - 0.01f, theta));
    }

    void zoom(float delta) {
        radius *= (1.0f - delta * 0.1f);
        radius = std::max(0.05f, radius);
    }

    void pan(float dx, float dy) {
        Vec3 eye = position();
        Vec3 f   = normalize(target - eye);
        Vec3 up{0.0f, 1.0f, 0.0f};
        Vec3 r   = normalize(cross(f, up));
        Vec3 u   = cross(r, f);
        float scale = radius * 0.001f;
        target = target - r * (dx * scale) + u * (dy * scale);
    }

    // Reset to fit a bounding sphere of the given radius centered at `center`
    // — mirrors the glTF Sample Viewer's default of orbiting the scene's
    // actual visual center rather than assuming it's at the origin.
    void fitBounds(float boundsRadius, Vec3 center = {0.0f, 0.0f, 0.0f}) {
        target = center;
        radius = boundsRadius * 2.5f;
        nearZ  = boundsRadius * 0.01f;
        farZ   = boundsRadius * 100.0f;
    }
};
