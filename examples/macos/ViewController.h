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
- (IBAction)loadEnvironmentMap:(id)sender;
- (IBAction)setBackgroundSolid:(id)sender;
- (IBAction)setBackgroundSkybox:(id)sender;
- (IBAction)setBackgroundSkyboxIBL:(id)sender;
- (IBAction)toggleFXAA:(id)sender;
- (IBAction)setSsaaOff:(id)sender;
- (IBAction)setSsaa15x:(id)sender;
- (IBAction)setSsaa20x:(id)sender;
@end
