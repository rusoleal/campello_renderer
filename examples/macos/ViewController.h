#pragma once
#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>

@interface ViewController : NSViewController <MTKViewDelegate>
- (void)loadURL:(NSURL *)url;
- (IBAction)toggleDebugMode:(id)sender;
- (IBAction)togglePunctualLights:(id)sender;
- (IBAction)toggleDefaultLight:(id)sender;
- (IBAction)setBackgroundDark:(id)sender;
- (IBAction)setBackgroundGray:(id)sender;
- (IBAction)setBackgroundLight:(id)sender;
- (IBAction)toggleSkybox:(id)sender;
- (IBAction)toggleIBL:(id)sender;
@end
