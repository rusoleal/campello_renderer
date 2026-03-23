// campello_renderer macOS example — ViewController
//
// Responsibilities:
//   - Creates an MTKView
//   - Creates a campello_gpu Device and campello_renderer Renderer
//   - Loads a GLTF/GLB file on request and delegates to Renderer
//   - Calls Renderer::render() each frame, passing the MTKView drawable
//   - Handles mouse/trackpad events for orbit/pan/zoom camera control

#import "ViewController.h"
#import "Camera.hpp"

#import <campello_gpu/device.hpp>
#import <campello_gpu/texture_view.hpp>
#import <campello_gpu/constants/pixel_format.hpp>

#import <campello_renderer/campello_renderer.hpp>

#import <gltf/gltf.hpp>

#import <simd/simd.h>

namespace gpu  = systems::leal::campello_gpu;
namespace gltf = systems::leal::gltf;
namespace rend = systems::leal::campello_renderer;

// ---------------------------------------------------------------------------
// ViewController implementation
// ---------------------------------------------------------------------------
@implementation ViewController {
    MTKView   *_metalView;

    std::shared_ptr<gpu::Device>    _device;
    std::shared_ptr<rend::Renderer> _renderer;

    Camera  _camera;
    NSPoint _lastMousePos;
    BOOL    _mouseDown;
    BOOL    _rightMouseDown;
}

// ---------------------------------------------------------------------------
// View lifecycle
// ---------------------------------------------------------------------------

- (void)loadView {
    id<MTLDevice> mtlDevice = MTLCreateSystemDefaultDevice();
    if (!mtlDevice) {
        NSLog(@"Metal is not supported on this device");
        self.view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 1280, 720)];
        return;
    }

    _metalView = [[MTKView alloc] initWithFrame:NSMakeRect(0, 0, 1280, 720)
                                         device:mtlDevice];
    _metalView.colorPixelFormat        = MTLPixelFormatBGRA8Unorm;
    _metalView.depthStencilPixelFormat = MTLPixelFormatInvalid; // renderer owns depth
    _metalView.clearColor              = MTLClearColorMake(0.08, 0.08, 0.10, 1.0);
    _metalView.delegate                = self;
    _metalView.preferredFramesPerSecond = 60;

    self.view = _metalView;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setupGPU];
}

// ---------------------------------------------------------------------------
// GPU + renderer initialisation
// ---------------------------------------------------------------------------

- (void)setupGPU {
    _device = gpu::Device::createDefaultDevice(nullptr);
    if (!_device) {
        NSLog(@"campello_gpu: failed to create device");
        return;
    }
    NSLog(@"device: %s  engine: %s",
          _device->getName().c_str(),
          gpu::Device::getEngineVersion().c_str());

    _renderer = std::make_shared<rend::Renderer>(_device);
    _renderer->createDefaultPipelines(gpu::PixelFormat::bgra8unorm);

    CGSize sz = _metalView.drawableSize;
    _renderer->resize((uint32_t)sz.width, (uint32_t)sz.height);
}

// ---------------------------------------------------------------------------
// File / URL loading
// ---------------------------------------------------------------------------

- (void)loadURL:(NSURL *)url {
    NSData *data = [NSData dataWithContentsOfURL:url options:0 error:nil];
    if (!data) {
        NSLog(@"campello_renderer_macos: could not read %@", url);
        return;
    }

    std::shared_ptr<gltf::GLTF> asset;
    NSString *ext = url.pathExtension.lowercaseString;

    if ([ext isEqualToString:@"glb"]) {
        asset = gltf::GLTF::loadGLB(
            reinterpret_cast<uint8_t *>(const_cast<void *>(data.bytes)),
            static_cast<uint64_t>(data.length));
    } else {
        asset = gltf::GLTF::loadGLTF(
            std::string(static_cast<const char *>(data.bytes),
                        static_cast<size_t>(data.length)));
    }

    if (!asset) {
        NSLog(@"campello_renderer_macos: failed to parse %@", url);
        return;
    }

    NSLog(@"campello_renderer_macos: loaded %@", url.lastPathComponent);

    _renderer->setAsset(asset);
    _camera.fitBounds(_renderer->getBoundsRadius());
}

// ---------------------------------------------------------------------------
// MTKViewDelegate — per-frame rendering
// ---------------------------------------------------------------------------

- (void)drawInMTKView:(MTKView *)view {
    if (!_renderer) return;

    id<CAMetalDrawable> drawable = view.currentDrawable;
    if (!drawable) return;

    // Pass the orbit camera matrices to the renderer each frame.
    CGSize sz = view.drawableSize;
    float aspect = (float)(sz.width / sz.height);
    simd_float4x4 viewMat = _camera.viewMatrix();
    simd_float4x4 projMat = _camera.projectionMatrix(aspect);
    _renderer->setCameraMatrices((const float *)&viewMat, (const float *)&projMat);

    // Render to the current drawable's texture.
    auto colorView = gpu::TextureView::fromNative((__bridge void *)drawable.texture);
    if (colorView) _renderer->render(colorView);

    [drawable present];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    if (_renderer) _renderer->resize((uint32_t)size.width, (uint32_t)size.height);
}

// ---------------------------------------------------------------------------
// Open file panel
// ---------------------------------------------------------------------------

- (IBAction)openDocument:(id)sender {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.allowsOtherFileTypes = YES;
    panel.title = @"Open GLTF / GLB file";

    [panel beginSheetModalForWindow:self.view.window
                  completionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            [self loadURL:panel.URL];
        }
    }];
}

// ---------------------------------------------------------------------------
// Mouse / trackpad events: orbit (left drag), pan (right drag / shift+left),
// zoom (scroll wheel)
// ---------------------------------------------------------------------------

- (void)mouseDown:(NSEvent *)event {
    _lastMousePos = [event locationInWindow];
    _mouseDown    = YES;
}
- (void)mouseUp:(NSEvent *)event   { _mouseDown = NO; }

- (void)mouseDragged:(NSEvent *)event {
    if (!_mouseDown) return;
    NSPoint pos = [event locationInWindow];
    float dx = (float)(pos.x - _lastMousePos.x);
    float dy = (float)(pos.y - _lastMousePos.y);
    _lastMousePos = pos;

    if (event.modifierFlags & NSEventModifierFlagShift) {
        _camera.pan(dx, dy);
    } else {
        _camera.orbit(-dx * 0.005f, dy * 0.005f);
    }
}

- (void)rightMouseDown:(NSEvent *)event {
    _lastMousePos   = [event locationInWindow];
    _rightMouseDown = YES;
}
- (void)rightMouseUp:(NSEvent *)event { _rightMouseDown = NO; }

- (void)rightMouseDragged:(NSEvent *)event {
    if (!_rightMouseDown) return;
    NSPoint pos = [event locationInWindow];
    float dx = (float)(pos.x - _lastMousePos.x);
    float dy = (float)(pos.y - _lastMousePos.y);
    _lastMousePos = pos;
    _camera.pan(dx, dy);
}

- (void)scrollWheel:(NSEvent *)event {
    float delta = (float)event.deltaY;
    _camera.zoom(event.hasPreciseScrollingDeltas ? delta * 0.2f : delta);
}

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)becomeFirstResponder  { return YES; }

@end
