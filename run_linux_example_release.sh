#!/bin/bash
# Run Linux example without Vulkan validation layers (Release builds).
# Build first with: ./build_linux_example.sh Release
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/build/linux/release/examples/linux/campello_renderer_linux"
"$APP" "$@"
