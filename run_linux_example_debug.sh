#!/bin/bash
# Run Linux example with Vulkan validation layers enabled (Debug builds).
# Build first with: ./build_linux_example.sh Debug
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/build/linux/debug/examples/linux/campello_renderer_linux"
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation "$APP" "$@"
