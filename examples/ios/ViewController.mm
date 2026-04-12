// campello_renderer iOS example — ViewController
//
// Responsibilities:
//   - Creates an MTKView fullscreen
//   - Creates a campello_gpu Device and campello_renderer Renderer
//   - Loads a GLTF/GLB file via UIDocumentPickerViewController
//   - Calls Renderer::render() each frame, passing the MTKView drawable
//   - Touch gesture controls:
//       1-finger pan  → orbit (azimuth / elevation)
//       2-finger pan  → camera pan
//       pinch         → zoom

#import "ViewController.h"
#import "Camera.hpp"

#import <campello_gpu/device.hpp>
#import <campello_gpu/texture_view.hpp>
#import <campello_gpu/constants/pixel_format.hpp>

#import <campello_renderer/campello_renderer.hpp>

#import <gltf/gltf.hpp>

#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <simd/simd.h>
#include <future>
#include <vector>

namespace gpu  = systems::leal::campello_gpu;
namespace gltf = systems::leal::gltf;
namespace rend = systems::leal::campello_renderer;

// ---------------------------------------------------------------------------
// ViewController implementation
// ---------------------------------------------------------------------------
@implementation ViewController {
    MTKView *_metalView;

    std::shared_ptr<gpu::Device>    _device;
    std::shared_ptr<rend::Renderer> _renderer;

    Camera _camera;

    // Gesture state
    CGFloat _lastPinchScale;
}

// ---------------------------------------------------------------------------
// View lifecycle
// ---------------------------------------------------------------------------

- (void)loadView {
    id<MTLDevice> mtlDevice = MTLCreateSystemDefaultDevice();
    if (!mtlDevice) {
        self.view = [[UIView alloc] initWithFrame:UIScreen.mainScreen.bounds];
        self.view.backgroundColor = [UIColor blackColor];
        return;
    }

    _metalView = [[MTKView alloc] initWithFrame:UIScreen.mainScreen.bounds
                                         device:mtlDevice];
    _metalView.colorPixelFormat        = MTLPixelFormatBGRA8Unorm;
    _metalView.depthStencilPixelFormat = MTLPixelFormatInvalid; // renderer owns depth
    _metalView.clearColor              = MTLClearColorMake(0.08, 0.08, 0.10, 1.0);
    _metalView.delegate                = self;
    _metalView.preferredFramesPerSecond = 60;
    _metalView.autoresizingMask        = UIViewAutoresizingFlexibleWidth
                                       | UIViewAutoresizingFlexibleHeight;

    self.view = _metalView;
}

- (void)viewDidLoad {
    [super viewDidLoad];

    self.title = @"campello_renderer";

    // Navigation bar buttons
    UIBarButtonItem *openBtn = [[UIBarButtonItem alloc]
        initWithTitle:@"Open"
                style:UIBarButtonItemStylePlain
               target:self
               action:@selector(openDocumentPicker)];

    UIBarButtonItem *debugBtn = [[UIBarButtonItem alloc]
        initWithTitle:@"Debug"
                style:UIBarButtonItemStylePlain
               target:self
               action:@selector(toggleDebugMode)];

    self.navigationItem.rightBarButtonItems = @[openBtn, debugBtn];

    [self setupGPU];
    [self setupGestures];
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
// Gesture recognisers
// ---------------------------------------------------------------------------

- (void)setupGestures {
    // 1-finger pan → orbit
    // maximumNumberOfTouches=1 already prevents this from firing during 2-finger gestures,
    // so no requireGestureRecognizerToFail needed (that would add a ~350ms delay).
    UIPanGestureRecognizer *orbitGesture =
        [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(handleOrbit:)];
    orbitGesture.maximumNumberOfTouches = 1;
    [_metalView addGestureRecognizer:orbitGesture];

    // 2-finger pan → camera pan
    UIPanGestureRecognizer *panGesture =
        [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(handlePan:)];
    panGesture.minimumNumberOfTouches = 2;
    panGesture.maximumNumberOfTouches = 2;
    [_metalView addGestureRecognizer:panGesture];

    // Pinch → zoom
    UIPinchGestureRecognizer *pinchGesture =
        [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(handlePinch:)];
    [_metalView addGestureRecognizer:pinchGesture];
}

- (void)handleOrbit:(UIPanGestureRecognizer *)gesture {
    CGPoint t = [gesture translationInView:_metalView];
    [gesture setTranslation:CGPointMake(0, 0) inView:_metalView];
    _camera.orbit((float)t.x * 0.005f, -(float)t.y * 0.005f);
}

- (void)handlePan:(UIPanGestureRecognizer *)gesture {
    CGPoint t = [gesture translationInView:_metalView];
    [gesture setTranslation:CGPointMake(0, 0) inView:_metalView];
    _camera.pan((float)t.x, -(float)t.y);
}

- (void)handlePinch:(UIPinchGestureRecognizer *)gesture {
    if (gesture.state == UIGestureRecognizerStateBegan) {
        _lastPinchScale = gesture.scale;
        return;
    }
    float delta = (float)(gesture.scale - _lastPinchScale);
    _lastPinchScale = gesture.scale;
    _camera.zoom(-delta * 5.0f);
}

// ---------------------------------------------------------------------------
// File loading — UIDocumentPickerViewController
// ---------------------------------------------------------------------------

- (void)openDocumentPicker {
    UTType *glbType  = [UTType typeWithFilenameExtension:@"glb"];
    UTType *gltfType = [UTType typeWithFilenameExtension:@"gltf"];
    NSArray<UTType *> *types = @[glbType ?: UTTypeData, gltfType ?: UTTypeData];

    UIDocumentPickerViewController *picker =
        [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:types
                                                                    asCopy:YES];
    picker.delegate             = (id<UIDocumentPickerDelegate>)self;
    picker.allowsMultipleSelection = NO;

    [self presentViewController:picker animated:YES completion:nil];
}

- (void)documentPicker:(UIDocumentPickerViewController *)controller
didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls {
    if (urls.count == 0) return;
    [self loadURL:urls.firstObject];
}

- (void)loadURL:(NSURL *)url {
    NSData *data = [NSData dataWithContentsOfURL:url options:0 error:nil];
    if (!data) {
        NSLog(@"campello_renderer_ios: could not read %@", url);
        return;
    }

    std::shared_ptr<gltf::GLTF> asset;
    NSString *ext = url.pathExtension.lowercaseString;

    if ([ext isEqualToString:@"glb"]) {
        asset = gltf::GLTF::loadGLB(
            reinterpret_cast<uint8_t *>(const_cast<void *>(data.bytes)),
            static_cast<uint64_t>(data.length));
    } else {
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
        NSLog(@"campello_renderer_ios: failed to parse %@", url);
        return;
    }

    NSLog(@"campello_renderer_ios: loaded %@", url.lastPathComponent);
    self.title = url.lastPathComponent;

    _renderer->setAsset(asset);
    _camera.fitBounds(_renderer->getBoundsRadius());
}

// ---------------------------------------------------------------------------
// Debug mode toggle
// ---------------------------------------------------------------------------

- (void)toggleDebugMode {
    if (!_renderer) return;
    bool enabled = !_renderer->isDebugModeEnabled();
    _renderer->setDebugMode(enabled);
    NSLog(@"Debug mode: %@", enabled ? @"ON" : @"OFF");
}

// ---------------------------------------------------------------------------
// MTKViewDelegate — per-frame rendering
// ---------------------------------------------------------------------------

- (void)drawInMTKView:(MTKView *)view {
    if (!_renderer) return;

    id<CAMetalDrawable> drawable = view.currentDrawable;
    if (!drawable || !drawable.texture) return;
    if (drawable.texture.pixelFormat == MTLPixelFormatInvalid) return;

    CGSize sz = view.drawableSize;
    if (sz.width == 0 || sz.height == 0) return;

    float aspect = (float)(sz.width / sz.height);
    simd_float4x4 viewMat = _camera.viewMatrix();
    simd_float4x4 projMat = _camera.projectionMatrix(aspect);
    _renderer->setCameraMatrices((const float *)&viewMat, (const float *)&projMat);

    auto colorView = gpu::TextureView::fromNative((__bridge void *)drawable.texture);
    if (colorView) _renderer->render(colorView);

    [drawable present];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    if (_renderer) _renderer->resize((uint32_t)size.width, (uint32_t)size.height);
}

@end
