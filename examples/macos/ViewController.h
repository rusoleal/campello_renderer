#pragma once
#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>

@interface ViewController : NSViewController <MTKViewDelegate>
- (void)loadURL:(NSURL *)url;
- (IBAction)toggleDebugMode:(id)sender;
@end
