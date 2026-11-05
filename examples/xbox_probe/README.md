# xbox_probe

A small UWP app that checks whether `campello_gpu`'s D3D12 backend and the renderer's embedded DXIL
can work on an Xbox Series X in Developer Mode. It repeats the backend's D3D12 calls and logs each
result. It doesn't link `campello_gpu` or `campello_renderer`, and it isn't part of the root CMake build.

## Requirements (Windows PC)

- Visual Studio with the **Universal Windows Platform development** workload and its C++ UWP tools
  (the project builds with that Visual Studio's default toolset; tested with v145 in VS 2026)
- Windows SDK 22621 or later (Xbox UWP development requires it)
- Developer Mode enabled in Windows settings

## Run it on the PC first

1. Open `xbox_probe.vcxproj` in Visual Studio and select `Debug` / `x64`.
2. Pick **Local Machine** as the target and press F5.

The window should turn green. This confirms the project builds and the code runs, but the results
say nothing about the Xbox.

## Run it on the Xbox

1. Connect the PC and console to the same network, wired if possible, and sign in on the console.
2. Open **Dev Home** on the console and note its IP address.
3. In the project properties, go to **Debugging**, set **Debugger to launch** to **Remote Machine**,
   **Machine Name** to the console's IP, and **Authentication Type** to **Universal (Unencrypted Protocol)**.
4. Press F5. On the first deploy, enter the PIN from Dev Home's **Show Visual Studio pin**.

Run it **twice**, because the two modes get different resources:

1. As deployed, which is normally **App** mode.
2. In **Game** mode: in Dev Home, highlight the package under *Games & apps*, press **View**, choose
   **View details**, set **App type** to **Game**, then run it again.

## Reading the result

| Screen | Meaning |
|---|---|
| Green | Everything the backend needs worked at feature level 11_0 |
| Amber | Device and present work, but a required check failed (search the log for `FAIL`) |
| Magenta | Running on the WARP software rasterizer, not the GPU |
| Splash screen stays | No D3D12 device or no swapchain (see the end of the log) |

## Getting the log

- In Visual Studio, copy the `[probe]` lines from the **Output** window (*Debug* pane).
- Otherwise, open Xbox Device Portal's File Explorer and go to
  `LocalAppData\campello.xboxprobe_*\LocalState\probe.log`.
