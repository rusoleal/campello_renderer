#pragma once
#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>

@interface ViewController : UIViewController <MTKViewDelegate>
- (void)loadURL:(NSURL *)url;
@end
