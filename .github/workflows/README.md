# CI/CD Workflows

This directory contains GitHub Actions workflows for building, testing, and releasing the campello_renderer library across multiple platforms.

## Workflows

### 1. Build Workflow (`build.yml`)

Triggered on every push to main/master/develop branches and pull requests.

**Platforms:**
- **Linux** (ubuntu-latest) - Builds Debug and Release configurations
- **Windows** (windows-latest) - Builds x64 Debug and Release
- **macOS** (macos-latest) - Builds library and example app
- **iOS** (macos-latest) - Cross-compiles for arm64 devices
- **Android** (ubuntu-latest) - Builds for all ABIs (arm64-v8a, armeabi-v7a, x86_64, x86)

**Features:**
- Runs unit tests on supported platforms
- Uploads build artifacts
- Uses Ninja generator where available for faster builds

### 2. Release Workflow (`release.yml`)

Triggered on version tags (e.g., `v1.0.0`) or manually via workflow_dispatch.

**Process:**
1. Builds release packages for all platforms
2. Creates platform-specific archives:
   - `campello_renderer-linux.tar.gz`
   - `campello_renderer-windows.zip`
   - `campello_renderer-macos.tar.gz`
   - `campello_renderer-ios.tar.gz`
   - `campello_renderer-android.tar.gz`
3. Creates a GitHub release with all packages attached

### 3. Code Quality Workflow (`code-quality.yml`)

Runs code quality checks on every push/PR.

**Checks:**
- **Formatting**: clang-format validation for C++ files
- **Static Analysis**: cppcheck for common issues
- **CMake**: cmake-format validation

## Requirements

### Android Builds
- JDK 17
- Android SDK with NDK 26.1.10909125
- CMake 3.22.1

### iOS Builds
- macOS runner (GitHub-hosted)
- Xcode with iOS SDK
- Minimum iOS deployment target: 14.0

### Windows Builds
- Visual Studio 2022 (MSVC)
- Windows SDK

## Usage

### Running CI on your fork

The workflows will automatically run on your fork when you push to the configured branches or create pull requests.

### Creating a Release

```bash
# Tag a release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

The release workflow will automatically create a GitHub release with all platform packages.

### Manual Build Trigger

You can manually trigger the build workflow from the GitHub Actions tab.

## Artifacts

Each workflow produces artifacts that can be downloaded from the GitHub Actions interface:

- Compiled libraries (.so, .dll, .dylib, .a)
- Example applications (where applicable)
- Android APK files

## Troubleshooting

### Android build failures
- Ensure the NDK version in the workflow matches your local setup
- Check that gradle wrapper is executable: `chmod +x gradlew`

### iOS build failures
- iOS builds require the `macos-latest` runner
- Check that CMAKE_OSX_DEPLOYMENT_TARGET is set appropriately

### Windows build failures
- Ensure CMake is generating for the correct architecture (-A x64)
- MSVC toolset must be initialized before CMake configuration
