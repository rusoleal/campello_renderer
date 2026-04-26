#import "AppDelegate.h"
#import "ViewController.h"

@implementation AppDelegate {
    NSWindow      *_window;
    ViewController *_viewController;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [self buildMenuBar];

    NSRect frame = NSMakeRect(0, 0, 1280, 720);
    NSWindowStyleMask style =
        NSWindowStyleMaskTitled          |
        NSWindowStyleMaskClosable        |
        NSWindowStyleMaskMiniaturizable  |
        NSWindowStyleMaskResizable;

    _window = [[NSWindow alloc] initWithContentRect:frame
                                          styleMask:style
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"campello_renderer — macOS";

    _viewController = [[ViewController alloc] init];
    _window.contentViewController = _viewController;

    [_window center];
    [_window makeKeyAndOrderFront:nil];

    // Load a file passed as command-line argument, e.g.:
    //   ./campello_renderer_macos /path/to/model.glb
    NSArray<NSString *> *args = [[NSProcessInfo processInfo] arguments];
    if (args.count > 1) {
        [_viewController loadURL:[NSURL fileURLWithPath:args[1]]];
    }
}

- (void)buildMenuBar {
    NSMenu *menuBar = [[NSMenu alloc] init];

    // App menu
    NSMenuItem *appItem = [[NSMenuItem alloc] init];
    [menuBar addItem:appItem];
    NSMenu *appMenu = [[NSMenu alloc] init];
    appItem.submenu = appMenu;
    [appMenu addItemWithTitle:@"Quit campello_renderer"
                       action:@selector(terminate:)
                keyEquivalent:@"q"];

    // File menu
    NSMenuItem *fileItem = [[NSMenuItem alloc] init];
    [menuBar addItem:fileItem];
    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    fileItem.submenu = fileMenu;
    [fileMenu addItemWithTitle:@"Open…"
                        action:@selector(openDocument:)
                 keyEquivalent:@"o"];

    // View menu
    NSMenuItem *viewItem = [[NSMenuItem alloc] init];
    [menuBar addItem:viewItem];
    NSMenu *viewMenu = [[NSMenu alloc] initWithTitle:@"View"];
    viewItem.submenu = viewMenu;
    NSMenuItem *debugItem = [[NSMenuItem alloc] initWithTitle:@"World Normal View"
                                                       action:@selector(toggleDebugMode:)
                                                keyEquivalent:@"d"];
    debugItem.keyEquivalentModifierMask = NSEventModifierFlagCommand;
    [viewMenu addItem:debugItem];

    // Lighting menu
    NSMenuItem *lightingItem = [[NSMenuItem alloc] init];
    [menuBar addItem:lightingItem];
    NSMenu *lightingMenu = [[NSMenu alloc] initWithTitle:@"Lighting"];
    lightingItem.submenu = lightingMenu;

    NSMenuItem *punctualItem = [[NSMenuItem alloc] initWithTitle:@"Punctual Lights"
                                                          action:@selector(togglePunctualLights:)
                                                   keyEquivalent:@"l"];
    punctualItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
    punctualItem.state = NSControlStateValueOn;
    [lightingMenu addItem:punctualItem];

    NSMenuItem *defaultItem = [[NSMenuItem alloc] initWithTitle:@"Default Light"
                                                         action:@selector(toggleDefaultLight:)
                                                  keyEquivalent:@"l"];
    defaultItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagOption;
    defaultItem.state = NSControlStateValueOn;
    [lightingMenu addItem:defaultItem];

    NSMenuItem *skyboxItem = [[NSMenuItem alloc] initWithTitle:@"Skybox"
                                                        action:@selector(toggleSkybox:)
                                                 keyEquivalent:@"b"];
    skyboxItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
    skyboxItem.state = NSControlStateValueOff;
    [lightingMenu addItem:skyboxItem];

    NSMenuItem *iblItem = [[NSMenuItem alloc] initWithTitle:@"Image-Based Lighting"
                                                     action:@selector(toggleIBL:)
                                              keyEquivalent:@"i"];
    iblItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
    iblItem.state = NSControlStateValueOn;
    [lightingMenu addItem:iblItem];

    [lightingMenu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *envItem = [[NSMenuItem alloc] initWithTitle:@"Load Environment Map…"
                                                     action:@selector(loadEnvironmentMap:)
                                              keyEquivalent:@"e"];
    envItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
    [lightingMenu addItem:envItem];

    NSMenuItem *bgModeItem = [[NSMenuItem alloc] init];
    [lightingMenu addItem:bgModeItem];
    NSMenu *bgModeMenu = [[NSMenu alloc] initWithTitle:@"Background Mode"];
    bgModeItem.submenu = bgModeMenu;
    [bgModeMenu addItemWithTitle:@"Solid Color"
                          action:@selector(setBackgroundSolid:)
                   keyEquivalent:@""];
    [bgModeMenu addItemWithTitle:@"Skybox"
                          action:@selector(setBackgroundSkybox:)
                   keyEquivalent:@""];
    [bgModeMenu addItemWithTitle:@"Skybox + IBL"
                          action:@selector(setBackgroundSkyboxIBL:)
                   keyEquivalent:@""];

    [lightingMenu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *fxaaItem = [[NSMenuItem alloc] initWithTitle:@"FXAA"
                                                      action:@selector(toggleFXAA:)
                                               keyEquivalent:@"a"];
    fxaaItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
    fxaaItem.state = NSControlStateValueOff;
    [lightingMenu addItem:fxaaItem];

    NSMenuItem *ssaaItem = [[NSMenuItem alloc] init];
    [lightingMenu addItem:ssaaItem];
    NSMenu *ssaaMenu = [[NSMenu alloc] initWithTitle:@"SSAA"];
    ssaaItem.submenu = ssaaMenu;
    [ssaaMenu addItemWithTitle:@"Off"
                        action:@selector(setSsaaOff:)
                 keyEquivalent:@""];
    [ssaaMenu addItemWithTitle:@"1.5×"
                        action:@selector(setSsaa15x:)
                 keyEquivalent:@""];
    [ssaaMenu addItemWithTitle:@"2.0×"
                        action:@selector(setSsaa20x:)
                 keyEquivalent:@""];

    [lightingMenu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *bgItem = [[NSMenuItem alloc] init];
    [lightingMenu addItem:bgItem];
    NSMenu *bgMenu = [[NSMenu alloc] initWithTitle:@"Background"];
    bgItem.submenu = bgMenu;
    [bgMenu addItemWithTitle:@"Dark"
                      action:@selector(setBackgroundDark:)
               keyEquivalent:@"1"];
    [bgMenu addItemWithTitle:@"Gray"
                      action:@selector(setBackgroundGray:)
               keyEquivalent:@"2"];
    [bgMenu addItemWithTitle:@"Light"
                      action:@selector(setBackgroundLight:)
               keyEquivalent:@"3"];

    [NSApp setMainMenu:menuBar];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end
