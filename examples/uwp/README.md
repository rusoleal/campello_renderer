# UWP example

Renders `DamagedHelmet.glb` with an HDR environment using `campello_renderer`, as a UWP app. The same
package runs on a Windows PC and on an Xbox in Developer Mode.

## Build

Requires Visual Studio with the **Universal Windows Platform development** workload and its C++ tools,
plus a Windows SDK. This command is for Visual Studio 2026; change `-G` for another version.

```powershell
cmake -S . -B build/uwp -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_SYSTEM_NAME=WindowsStore -DCMAKE_SYSTEM_VERSION=10.0 `
  -DCMAKE_VS_WINDOWS_TARGET_PLATFORM_MIN_VERSION=10.0.19041.0 `
  -DCAMPELLO_RENDERER_BUILD_UWP_EXAMPLE=ON -DCAMPELLO_RENDERER_ECS_SUPPORT=OFF `
  -DFETCHCONTENT_SOURCE_DIR_EXTERN_CAMPELLO_GPU=<path to your campello_gpu clone>
cmake --build build/uwp --config Debug --target campello_renderer_uwp
```

`campello_gpu` needs `createDefaultDeviceForCoreWindow()`, so point CMake at a clone that has it until
a release includes it.

The build writes an unsigned package to
`build/uwp/examples/uwp/AppPackages/campello_renderer_uwp/campello_renderer_uwp_1.0.0.0_x64_Debug_Test/`.

## Run on the PC

Developer Mode must be on. Unpack the package and register the result:

```powershell
makeappx unpack /o /p <package>.appx /d build/uwp/layout
Add-AppxPackage -Register build/uwp/layout/AppxManifest.xml
```

The package also needs a recent `Microsoft.VCLibs.140.00.Debug`. If registering asks for it, install
`Dependencies\x64\Microsoft.VCLibs.x64.Debug.14.00.appx` from the package folder.

## Run on the Xbox

1. Sign the package with a certificate whose subject is `CN=campello`, matching the manifest's Publisher.
   A self-signed certificate works.
2. Install the signed package and the VCLibs dependency through Xbox Device Portal.
3. Start it from Dev Home. To get more memory, set **App type** to **Game**: the limit rises from 1 GB
   to 5 GB.

## Controls

| Gamepad | Action |
|---|---|
| Left stick | Orbit |
| Triggers | Zoom |
| Right bumper | Pause the automatic orbit |
| A | Next debug view |
| X | FXAA on or off |
| Y | IBL and skybox on or off |

## Logs

There's no console on UWP, so the app writes to its `LocalState` folder:

| File | Contents |
|---|---|
| `app.log` | Startup timings, frame rate, control changes and crash details |
| `stdout.log`, `stderr.log` | Output from the campello libraries |

On a PC the folder is `%LOCALAPPDATA%\Packages\campello.rendereruwp_*\LocalState`. On an Xbox, use Device
Portal's File Explorer. The app keeps `stdout.log` and `stderr.log` open, so stop it before reading them.
