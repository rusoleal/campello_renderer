// campello_renderer shader program interface, WGSL (WebGPU).
// Reserved: the renderer has no WebGPU mesh path yet (programmable pipelines
// phase 7). The intended contract, matching the other backends:
//   @location(0..11)  vertex attributes = VertexSemantic
//   @location(16..19) mvp columns, @location(24..27) model columns
//                     (per-instance vertex buffer after the attributes)
// WGSL has no #include: the renderer will prepend this file to app WGSL.
