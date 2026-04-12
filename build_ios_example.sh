#!/bin/bash
# Build the campello_renderer iOS example and launch it in the iOS Simulator.
#
# Usage:
#   ./build_ios_example.sh                       # default simulator (iPhone 16 Pro)
#   ./build_ios_example.sh "iPhone 17 Pro"       # specific simulator name
#   CAMPELLO_NO_LAUNCH=1 ./build_ios_example.sh  # build only, don't launch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/ios-example"
# On Apple Silicon Macs the simulator runs arm64; on Intel Macs it runs x86_64.
HOST_ARCH=$(uname -m)
if [ "$HOST_ARCH" = "x86_64" ]; then
    SIM_ARCH="x86_64"
else
    SIM_ARCH="arm64"
fi

SIMULATOR="${1:-iPhone 17 Pro}"
BUNDLE_ID="systems.leal.campello-renderer-ios"

echo "=== campello_renderer iOS example ==="
echo "  Simulator : $SIMULATOR"
echo "  Build dir : $BUILD_DIR"
echo ""

# ---------------------------------------------------------------------------
# Resolve simulator UDID up front
# ---------------------------------------------------------------------------
DEVICE_UDID=$(xcrun simctl list devices available -j \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
name = '$SIMULATOR'
for runtime, devices in data.get('devices', {}).items():
    for d in devices:
        if d.get('name') == name and d.get('isAvailable', False):
            print(d['udid'])
            exit()
" 2>/dev/null)

if [ -z "$DEVICE_UDID" ]; then
    echo "ERROR: Simulator '$SIMULATOR' not found. Available simulators:"
    xcrun simctl list devices available | grep "iPhone"
    exit 1
fi
echo "  UDID      : $DEVICE_UDID"
echo ""

# ---------------------------------------------------------------------------
# 1. CMake configure — explicitly target iphonesimulator sysroot
# ---------------------------------------------------------------------------
echo "--- Configuring ---"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCAMPELLO_RENDERER_BUILD_IOS_EXAMPLE=ON \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT=iphonesimulator \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
    -DCMAKE_OSX_ARCHITECTURES=$SIM_ARCH \
    -GXcode

# ---------------------------------------------------------------------------
# 2. Build for iOS simulator using UDID destination (no code signing)
# ---------------------------------------------------------------------------
echo "--- Building ---"
xcodebuild \
    -project "$BUILD_DIR/campello_renderer.xcodeproj" \
    -scheme campello_renderer_ios \
    -configuration Debug \
    -sdk iphonesimulator \
    -destination "generic/platform=iOS Simulator" \
    CODE_SIGNING_ALLOWED=NO \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGN_IDENTITY="" \
    build

APP_PATH=$(find "$BUILD_DIR" -name "campello_renderer_ios.app" \
    -path "*iphonesimulator*" | head -1)

if [ -z "$APP_PATH" ]; then
    # Also try without the iphonesimulator path filter (cmake may use a different layout)
    APP_PATH=$(find "$BUILD_DIR" -name "campello_renderer_ios.app" | head -1)
fi

if [ -z "$APP_PATH" ]; then
    echo "ERROR: Could not find campello_renderer_ios.app in $BUILD_DIR"
    exit 1
fi
echo "  App: $APP_PATH"

# ---------------------------------------------------------------------------
# 3. Boot simulator and install + launch app
# ---------------------------------------------------------------------------
if [ "${CAMPELLO_NO_LAUNCH:-0}" = "1" ]; then
    echo "--- Build complete (launch skipped) ---"
    exit 0
fi

echo "--- Booting simulator: $SIMULATOR ($DEVICE_UDID) ---"
STATUS=$(xcrun simctl list devices -j | python3 -c "
import sys, json
data = json.load(sys.stdin)
for runtime, devices in data.get('devices', {}).items():
    for d in devices:
        if d.get('udid') == '$DEVICE_UDID':
            print(d.get('state', 'Shutdown'))
            exit()
")

if [ "$STATUS" != "Booted" ]; then
    echo "  Booting..."
    xcrun simctl boot "$DEVICE_UDID"
fi

# Open Simulator.app so the window appears
open -a Simulator

echo "--- Installing ---"
xcrun simctl install "$DEVICE_UDID" "$APP_PATH"

echo "--- Launching ---"
xcrun simctl launch "$DEVICE_UDID" "$BUNDLE_ID"

echo ""
echo "Done. Tap 'Open' in the app to load a .glb / .gltf file."
echo "To copy a GLB into the simulator Files app:"
echo "  xcrun simctl addmedia $DEVICE_UDID /path/to/model.glb"
