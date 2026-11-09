#pragma once

// Frustum culling helpers shared by render_system() (ecs.hpp) and usable on
// their own: no campello_core dependency.

#include <campello_renderer/campello_renderer.hpp>
#include <vector_math/vector_math.hpp>

namespace systems::leal::campello_renderer {

/// True when a mesh's local-space bounds lie entirely outside the view
/// frustum, i.e. when `clipFromWorld * world` (projection * view * world,
/// row-major, column vectors) maps the whole box outside one clip plane.
/// The test runs in the mesh's local space, so it stays exact under any
/// rotation or scale in `world`. Conservative: the near plane is z >= -w
/// (looser than a [0, 1] depth range), so nothing visible is ever culled.
/// Meshes without bounds are never outside.
inline bool isOutsideFrustum(const systems::leal::vector_math::Matrix4<double>& clipFromWorld,
                             const systems::leal::vector_math::Matrix4<double>& world,
                             const GpuMesh& mesh)
{
    if (!mesh.hasBounds) return false;
    const auto m = clipFromWorld * world;
    const double* r = m.data;  // row i at r[i*4 + 0..3]
    // Planes as (a, b, c, d) with a*x + b*y + c*z + d >= 0 inside:
    // left, right, bottom, top, near, far (row3 +/- row0/1/2).
    const double planes[6][4] = {
        {r[12] + r[0], r[13] + r[1], r[14] + r[2],  r[15] + r[3]},
        {r[12] - r[0], r[13] - r[1], r[14] - r[2],  r[15] - r[3]},
        {r[12] + r[4], r[13] + r[5], r[14] + r[6],  r[15] + r[7]},
        {r[12] - r[4], r[13] - r[5], r[14] - r[6],  r[15] - r[7]},
        {r[12] + r[8], r[13] + r[9], r[14] + r[10], r[15] + r[11]},
        {r[12] - r[8], r[13] - r[9], r[14] - r[10], r[15] - r[11]},
    };
    for (const auto& p : planes) {
        // The box corner furthest along the plane normal: if even it is
        // outside, the whole box is.
        const double x = p[0] >= 0.0 ? mesh.boundsMax[0] : mesh.boundsMin[0];
        const double y = p[1] >= 0.0 ? mesh.boundsMax[1] : mesh.boundsMin[1];
        const double z = p[2] >= 0.0 ? mesh.boundsMax[2] : mesh.boundsMin[2];
        if (p[0] * x + p[1] * y + p[2] * z + p[3] < 0.0) return true;
    }
    return false;
}

} // namespace systems::leal::campello_renderer
