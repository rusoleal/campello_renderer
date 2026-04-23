#!/bin/bash
# Run macOS example without Metal API Validation (Release builds).
# Build first with: ./build_macos_example.sh Release
APP="build/macos/release/examples/macos/campello_renderer_macos.app"
"$APP/Contents/MacOS/campello_renderer_macos"
