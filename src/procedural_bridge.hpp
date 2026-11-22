#pragma once

// KHR_texture_procedurals: gltf's parsed graphs as campello_procedural
// graphs, and the images their `image` nodes read from the asset. Internal
// to the renderer (campello_procedural is linked privately).

#include <campello_procedural/campello_procedural.hpp>
#include <gltf/gltf.hpp>

namespace systems::leal::campello_renderer::procedural {

/// The graph, converted (types by name, values, connections, UI data).
campello_procedural::Graph toGraph(const systems::leal::gltf::ProceduralGraph& graph);

/// Supplies `image` nodes from the asset's images (by URI) or the file
/// system, decoded to float RGBA as stored (not linearized).
campello_procedural::ImageResolver imageResolver(const systems::leal::gltf::GLTF& asset);

/// Whether this thread may start threads and wait for them. Not the
/// browser's main thread: a Web Worker beyond Emscripten's pre-started pool
/// only starts once the main thread yields, so join() would wait forever (a
/// KHR_texture_procedurals model froze the page on machines with more cores
/// than the pool). Work runs single-threaded there instead.
bool canWaitForThreads();

}  // namespace systems::leal::campello_renderer::procedural
