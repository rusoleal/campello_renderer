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
#include <future>
#include <vector>

namespace gpu  = systems::leal::campello_gpu;
namespace gltf = systems::leal::gltf;
namespace rend = systems::leal::campello_renderer;

// ---------------------------------------------------------------------------
// Transparent drag-drop overlay view
// ---------------------------------------------------------------------------
@interface DragDropView : NSView
@property (nonatomic, assign) ViewController *viewController;
@end

@implementation DragDropView

- (instancetype)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        [self registerForDraggedTypes:@[NSPasteboardTypeFileURL]];
    }
    return self;
}

- (void)viewDidMoveToWindow {
    [super viewDidMoveToWindow];
    if (self.window) {
        [self registerForDraggedTypes:@[NSPasteboardTypeFileURL]];
    }
}

- (BOOL)isModelFileURL:(NSURL *)url {
    NSString *ext = url.pathExtension.lowercaseString;
    return [ext isEqualToString:@"glb"] || [ext isEqualToString:@"gltf"];
}

- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender {
    NSPasteboard *pb = sender.draggingPasteboard;
    NSArray<NSURL *> *urls = [pb readObjectsForClasses:@[[NSURL class]]
                                               options:@{NSPasteboardURLReadingFileURLsOnlyKey: @YES}];
    for (NSURL *url in urls) {
        if ([self isModelFileURL:url]) {
            return NSDragOperationCopy;
        }
    }
    return NSDragOperationNone;
}

- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender {
    NSPasteboard *pb = sender.draggingPasteboard;
    NSArray<NSURL *> *urls = [pb readObjectsForClasses:@[[NSURL class]]
                                               options:@{NSPasteboardURLReadingFileURLsOnlyKey: @YES}];
    for (NSURL *url in urls) {
        if ([self isModelFileURL:url]) {
            [self.viewController loadURL:url];
            return YES;
        }
    }
    return NO;
}

@end

static NSString *ViewModeName(rend::ViewMode mode) {
    switch (mode) {
        case rend::ViewMode::normal: return @"normal";
        case rend::ViewMode::worldNormal: return @"worldNormal";
        case rend::ViewMode::baseColor: return @"baseColor";
        case rend::ViewMode::metallic: return @"metallic";
        case rend::ViewMode::roughness: return @"roughness";
        case rend::ViewMode::occlusion: return @"occlusion";
        case rend::ViewMode::emissive: return @"emissive";
        case rend::ViewMode::alpha: return @"alpha";
        case rend::ViewMode::uv0: return @"uv0";
        case rend::ViewMode::specularFactor: return @"specularFactor";
        case rend::ViewMode::specularColor: return @"specularColor";
        case rend::ViewMode::sheenColor: return @"sheenColor";
        case rend::ViewMode::sheenRoughness: return @"sheenRoughness";
        case rend::ViewMode::clearcoat: return @"clearcoat";
        case rend::ViewMode::clearcoatRoughness: return @"clearcoatRoughness";
        case rend::ViewMode::clearcoatNormal: return @"clearcoatNormal";
        case rend::ViewMode::transmission: return @"transmission";
        case rend::ViewMode::environment: return @"environment";
        case rend::ViewMode::iridescence: return @"iridescence";
        case rend::ViewMode::anisotropy: return @"anisotropy";
        case rend::ViewMode::dispersion: return @"dispersion";
    }
}

static bool ViewModeForKey(NSString *key, rend::ViewMode &outMode) {
    if (!key || key.length != 1) return false;
    unichar ch = [[key lowercaseString] characterAtIndex:0];
    switch (ch) {
        case '0': outMode = rend::ViewMode::normal; return true;
        case '1': outMode = rend::ViewMode::worldNormal; return true;
        case '2': outMode = rend::ViewMode::baseColor; return true;
        case '3': outMode = rend::ViewMode::metallic; return true;
        case '4': outMode = rend::ViewMode::roughness; return true;
        case '5': outMode = rend::ViewMode::occlusion; return true;
        case '6': outMode = rend::ViewMode::emissive; return true;
        case '7': outMode = rend::ViewMode::alpha; return true;
        case '8': outMode = rend::ViewMode::uv0; return true;
        case '9': outMode = rend::ViewMode::specularFactor; return true;
        case 'q': outMode = rend::ViewMode::specularColor; return true;
        case 'w': outMode = rend::ViewMode::sheenColor; return true;
        case 'e': outMode = rend::ViewMode::sheenRoughness; return true;
        case 'r': outMode = rend::ViewMode::clearcoat; return true;
        case 't': outMode = rend::ViewMode::clearcoatRoughness; return true;
        case 'y': outMode = rend::ViewMode::clearcoatNormal; return true;
        case 'u': outMode = rend::ViewMode::transmission; return true;
        case 'i': outMode = rend::ViewMode::environment; return true;
        case 'o': outMode = rend::ViewMode::iridescence; return true;
        case 'p': outMode = rend::ViewMode::anisotropy; return true;
        case 'z': outMode = rend::ViewMode::dispersion; return true;
        default: return false;
    }
}

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
    BOOL    _debugModeEnabled;
    CGSize  _lastDrawableSize;

    dispatch_queue_t _renderQueue;
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

    // Container view: self.view holds both MTKView and the drag-drop overlay as siblings.
    NSView *container = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 1280, 720)];
    container.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.view = container;

    _metalView = [[MTKView alloc] initWithFrame:container.bounds
                                         device:mtlDevice];
    _metalView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _metalView.colorPixelFormat        = MTLPixelFormatBGRA8Unorm;
    _metalView.depthStencilPixelFormat = MTLPixelFormatInvalid; // renderer owns depth
    _metalView.clearColor              = MTLClearColorMake(0.08, 0.08, 0.10, 1.0);
    _metalView.delegate                = self;
    _metalView.preferredFramesPerSecond = 60;
    [container addSubview:_metalView];

    // Transparent overlay view for drag-and-drop.
    DragDropView *dragView = [[DragDropView alloc] initWithFrame:container.bounds];
    dragView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    dragView.viewController = self;
    [container addSubview:dragView];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setupGPU];
}

- (void)viewDidAppear {
    [super viewDidAppear];
    [self.view.window makeFirstResponder:self];
}

// ---------------------------------------------------------------------------
// GPU + renderer initialisation
// ---------------------------------------------------------------------------

- (void)setupGPU {
    _renderQueue = dispatch_queue_create("com.campello.render", DISPATCH_QUEUE_SERIAL);

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
    dispatch_sync(_renderQueue, ^{
        _renderer->resize((uint32_t)sz.width, (uint32_t)sz.height);
    });
    _lastDrawableSize = sz;
    
    _debugModeEnabled = NO;

    NSLog(@"ViewMode hotkeys: 0 normal, 1 worldNormal, 2 baseColor, 3 metallic, 4 roughness, 5 occlusion, 6 emissive, 7 alpha, 8 uv0, 9 specularFactor, q specularColor, w sheenColor, e sheenRoughness, r clearcoat, t clearcoatRoughness, y clearcoatNormal, u transmission, i environment");
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
        // Get the base directory of the GLTF file for resolving relative URIs.
        NSString *baseDir = url.path.stringByDeletingLastPathComponent;
        asset = gltf::GLTF::loadGLTF(
            std::string(static_cast<const char *>(data.bytes),
                        static_cast<size_t>(data.length)),
            [baseDir](const std::string &uri) -> std::future<std::vector<uint8_t>> {
                return std::async(std::launch::deferred, [baseDir, uri]() {
                    NSString *nsUri = [NSString stringWithUTF8String:uri.c_str()];
                    NSString *fullPath = [baseDir stringByAppendingPathComponent:nsUri];
                    NSData *fileData = [NSData dataWithContentsOfFile:fullPath];
                    std::vector<uint8_t> result;
                    if (fileData) {
                        const uint8_t *bytes = (const uint8_t *)fileData.bytes;
                        result.assign(bytes, bytes + fileData.length);
                    }
                    return result;
                });
            });
    }

    if (!asset) {
        NSLog(@"campello_renderer_macos: failed to parse %@", url);
        return;
    }

    NSLog(@"campello_renderer_macos: loaded %@", url.lastPathComponent);

    dispatch_sync(_renderQueue, ^{
        _renderer->setAsset(asset);
        _camera.fitBounds(_renderer->getBoundsRadius());

        uint32_t animCount = _renderer->getAnimationCount();
        NSLog(@"Animations found: %u", animCount);
        for (uint32_t i = 0; i < animCount; ++i) {
            NSLog(@"  Animation %u: %s (duration: %.2fs)", 
                  i, 
                  _renderer->getAnimationName(i).c_str(),
                  _renderer->getAnimationDuration(i));
        }

        if (animCount > 0) {
            _renderer->playAnimation(0);
            NSLog(@"Auto-playing animation: %s", _renderer->getAnimationName(0).c_str());
        }
    });
}

// ---------------------------------------------------------------------------
// MTKViewDelegate — per-frame rendering
// ---------------------------------------------------------------------------

- (void)drawInMTKView:(MTKView *)view {
    if (!_renderer || !_renderQueue) return;

    id<CAMetalDrawable> drawable = view.currentDrawable;
    if (!drawable || !drawable.texture) return;
    if (drawable.texture.pixelFormat == MTLPixelFormatInvalid) return;

    dispatch_sync(_renderQueue, ^{
        // Update animations (call with 1/60s delta time for 60fps).
        _renderer->update(1.0 / 60.0);

        // Use the ACTUAL drawable texture size for rendering.
        CGSize sz = CGSizeMake(drawable.texture.width, drawable.texture.height);
        if (sz.width == 0 || sz.height == 0) return;

        BOOL didResize = NO;
        if (sz.width != _lastDrawableSize.width || sz.height != _lastDrawableSize.height) {
            NSLog(@"[RESIZE] drawable %.0fx%.0f → last %.0fx%.0f  (calling resize %dx%d)",
                  sz.width, sz.height, _lastDrawableSize.width, _lastDrawableSize.height,
                  (int)sz.width, (int)sz.height);
            _renderer->resize((uint32_t)sz.width, (uint32_t)sz.height);
            _lastDrawableSize = sz;
            didResize = YES;
        }
        
        float aspect = (float)(sz.width / sz.height);
        simd_float4x4 viewMat = _camera.viewMatrix();
        simd_float4x4 projMat = _camera.projectionMatrix(aspect);
        _renderer->setCameraMatrices((const float *)&viewMat, (const float *)&projMat);

        // Render to the current drawable's texture.
        auto colorView = gpu::TextureView::fromNative((__bridge void *)drawable.texture);
        if (colorView) _renderer->render(colorView);

        if (didResize) {
            NSLog(@"[RESIZE] frame submitted after resize to %dx%d", (int)sz.width, (int)sz.height);
        }

        [drawable present];
    });
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    // Defer the actual resize to drawInMTKView: where we can match the
    // depth texture size to the real drawable texture size.  On macOS
    // the drawable pool can lag behind drawableSizeWillChange: by a
    // frame, so resizing here creates a depth/color attachment mismatch.
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

- (IBAction)toggleDebugMode:(id)sender {
    if (!_renderer || !_renderQueue) return;

    dispatch_sync(_renderQueue, ^{
        rend::ViewMode nextMode = _renderer->getViewMode() == rend::ViewMode::worldNormal
            ? rend::ViewMode::normal
            : rend::ViewMode::worldNormal;
        _renderer->setViewMode(nextMode);
        
        // Update menu item state
        if ([sender isKindOfClass:[NSMenuItem class]]) {
            NSMenuItem *item = (NSMenuItem *)sender;
            item.state = (_renderer->getViewMode() == rend::ViewMode::worldNormal)
                ? NSControlStateValueOn
                : NSControlStateValueOff;
        }
    });
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

- (void)keyDown:(NSEvent *)event {
    rend::ViewMode mode;
    if (ViewModeForKey(event.charactersIgnoringModifiers, mode)) {
        [self applyViewMode:mode];
        return;
    }
    [super keyDown:event];
}

- (void)applyViewMode:(rend::ViewMode)mode {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setViewMode(mode);
        _debugModeEnabled = (mode == rend::ViewMode::worldNormal);
    });
    NSLog(@"View mode: %@", ViewModeName(mode));
}

// ---------------------------------------------------------------------------
// Lighting controls
// ---------------------------------------------------------------------------

- (IBAction)togglePunctualLights:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        bool enabled = !_renderer->isPunctualLightsEnabled();
        _renderer->setPunctualLightsEnabled(enabled);
        NSLog(@"Punctual lights: %@", enabled ? @"ON" : @"OFF");
    });
    if ([sender isKindOfClass:[NSMenuItem class]]) {
        NSMenuItem *item = (NSMenuItem *)sender;
        item.state = _renderer->isPunctualLightsEnabled() ? NSControlStateValueOn : NSControlStateValueOff;
    }
}

- (IBAction)toggleDefaultLight:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        bool enabled = !_renderer->isDefaultLightEnabled();
        _renderer->setDefaultLightEnabled(enabled);
        NSLog(@"Default light: %@", enabled ? @"ON" : @"OFF");
    });
    if ([sender isKindOfClass:[NSMenuItem class]]) {
        NSMenuItem *item = (NSMenuItem *)sender;
        item.state = _renderer->isDefaultLightEnabled() ? NSControlStateValueOn : NSControlStateValueOff;
    }
}

- (IBAction)setBackgroundDark:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    });
}

- (IBAction)setBackgroundGray:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setClearColor(0.35f, 0.35f, 0.38f, 1.0f);
    });
}

- (IBAction)setBackgroundLight:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setClearColor(0.78f, 0.82f, 0.88f, 1.0f);
    });
}

- (IBAction)toggleSkybox:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        bool enabled = !_renderer->isSkyboxEnabled();
        _renderer->setSkyboxEnabled(enabled);
        NSLog(@"Skybox: %@", enabled ? @"ON" : @"OFF");
    });
    if ([sender isKindOfClass:[NSMenuItem class]]) {
        NSMenuItem *item = (NSMenuItem *)sender;
        item.state = _renderer->isSkyboxEnabled() ? NSControlStateValueOn : NSControlStateValueOff;
    }
}

- (IBAction)toggleIBL:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        bool enabled = !_renderer->isIBLEnabled();
        _renderer->setIBLEnabled(enabled);
        NSLog(@"IBL: %@", enabled ? @"ON" : @"OFF");
    });
    if ([sender isKindOfClass:[NSMenuItem class]]) {
        NSMenuItem *item = (NSMenuItem *)sender;
        item.state = _renderer->isIBLEnabled() ? NSControlStateValueOn : NSControlStateValueOff;
    }
}

- (IBAction)loadEnvironmentMap:(id)sender {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.allowsOtherFileTypes = YES;
    panel.title = @"Load Environment Map";
    panel.allowedFileTypes = @[@"hdr", @"exr", @"png", @"jpg", @"jpeg", @"webp"];

    [panel beginSheetModalForWindow:self.view.window
                  completionHandler:^(NSModalResponse result) {
        if (result != NSModalResponseOK) return;
        NSURL *url = panel.URL;
        if (!url) return;

        NSString *path = url.path;
        std::string cppPath = std::string([path UTF8String]);

        dispatch_sync(_renderQueue, ^{
            auto tex = _renderer->loadEquirectangularEnvironmentMap(cppPath, 0);
            if (tex) {
                _renderer->setEnvironmentMap(tex);
                _renderer->setSkyboxEnabled(true);
                _renderer->setIBLEnabled(true);
                NSLog(@"Environment map loaded: %@", path.lastPathComponent);
            } else {
                NSLog(@"Failed to load environment map: %@", path);
            }
        });
    }];
}

- (IBAction)setBackgroundSolid:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setSkyboxEnabled(false);
        _renderer->setIBLEnabled(false);
        _renderer->setClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    });
    NSLog(@"Background: Solid Color (Dark)");
}

- (IBAction)setBackgroundSkybox:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setSkyboxEnabled(true);
        _renderer->setIBLEnabled(false);
    });
    NSLog(@"Background: Skybox");
}

- (IBAction)setBackgroundSkyboxIBL:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setSkyboxEnabled(true);
        _renderer->setIBLEnabled(true);
    });
    NSLog(@"Background: Skybox + IBL");
}

- (IBAction)toggleFXAA:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        bool enabled = !_renderer->isFxaaEnabled();
        _renderer->setFxaaEnabled(enabled);
        NSLog(@"FXAA: %@", enabled ? @"ON" : @"OFF");
    });
    if ([sender isKindOfClass:[NSMenuItem class]]) {
        NSMenuItem *item = (NSMenuItem *)sender;
        item.state = _renderer->isFxaaEnabled() ? NSControlStateValueOn : NSControlStateValueOff;
    }
}

- (IBAction)setSsaaOff:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setSsaaScale(1.0f);
        NSLog(@"SSAA: OFF");
    });
}

- (IBAction)setSsaa15x:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setSsaaScale(1.5f);
        NSLog(@"SSAA: 1.5×");
    });
}

- (IBAction)setSsaa20x:(id)sender {
    if (!_renderer || !_renderQueue) return;
    dispatch_sync(_renderQueue, ^{
        _renderer->setSsaaScale(2.0f);
        NSLog(@"SSAA: 2.0×");
    });
}

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)becomeFirstResponder  { return YES; }

@end
