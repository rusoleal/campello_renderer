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
    NSMenuItem *debugItem = [[NSMenuItem alloc] initWithTitle:@"Debug Mode"
                                                       action:@selector(toggleDebugMode:)
                                                keyEquivalent:@"d"];
    debugItem.keyEquivalentModifierMask = NSEventModifierFlagCommand;
    [viewMenu addItem:debugItem];

    [NSApp setMainMenu:menuBar];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end
