#import <AppKit/AppKit.h>
#import <ScriptingBridge/ScriptingBridge.h>
#import <unistd.h>

// Only the front window and its active pane are addressed. These declarations
// mirror the small read-only subset of the installed Finder/QSpace dictionaries.
@protocol SAFrontFinderApplication
- (SBElementArray *)FinderWindows;
@end
@protocol SAFrontFinderWindow
- (NSInteger)id;
- (BOOL)visible;
- (BOOL)collapsed;
- (SBObject *)target;
@end
@protocol SAFrontFinderItem
- (NSString *)URL;
@end
@protocol SAFrontQSpaceApplication
- (SBElementArray *)windows;
@end
@protocol SAFrontQSpaceWindow
- (NSString *)id;
- (NSRect)bounds;
- (SBObject *)activatedPane;
@end
@protocol SAFrontQSpacePane
- (NSString *)id;
- (SBObject *)rootItem;
@end
@protocol SAFrontQSpaceItem
- (NSString *)urlstr;
@end

@interface SAReadErrors : NSObject <SBApplicationDelegate>
@property(nonatomic) NSInteger code;
@end
@implementation SAReadErrors
- (id)eventDidFail:(const AppleEvent *)event withError:(NSError *)error {
    self.code = error.code;
    return nil;
}
@end

static BOOL isFront(NSString *bundle, pid_t pid) {
    NSRunningApplication *front = NSWorkspace.sharedWorkspace.frontmostApplication;
    return front.processIdentifier == pid && [front.bundleIdentifier isEqualToString:bundle];
}

static NSDictionary *emptySnapshot(void) { return @{ @"windows": @[] }; }

static NSDictionary *readFinder(SBApplication *app, SAReadErrors *errors, NSString *bundle, pid_t pid) {
    SBObject<SAFrontFinderWindow> *window = [(id<SAFrontFinderApplication>)app FinderWindows][0];
    if (!isFront(bundle, pid)) return nil;
    BOOL visible = [window visible];
    if (errors.code == -1728) { errors.code = 0; return emptySnapshot(); }
    if (errors.code != 0 || !isFront(bundle, pid)) return nil;
    BOOL collapsed = [window collapsed];
    if (errors.code != 0 || !isFront(bundle, pid)) return nil;
    if (!visible || collapsed) return emptySnapshot();
    NSInteger windowID = [window id];
    if (errors.code != 0 || !isFront(bundle, pid)) return nil;
    NSString *url = [(id<SAFrontFinderItem>)[window target] URL];
    // A missing target is a valid visible window, but a permission/read error is not.
    if (errors.code != 0 && errors.code != -1728) return nil;
    if (!isFront(bundle, pid)) return nil;
    errors.code = 0;
    return @{ @"windows": @[ @{ @"id": [NSString stringWithFormat:@"%ld", (long)windowID],
                                @"front": @YES, @"url": url ?: [NSNull null] } ] };
}

static NSDictionary *readQSpace(SBApplication *app, SAReadErrors *errors, NSString *bundle, pid_t pid) {
    SBObject<SAFrontQSpaceWindow> *window = [(id<SAFrontQSpaceApplication>)app windows][0];
    if (!isFront(bundle, pid)) return nil;
    NSString *windowID = [window id];
    if (errors.code == -1728) { errors.code = 0; return emptySnapshot(); }
    if (errors.code != 0 || !isFront(bundle, pid)) return nil;
    NSRect bounds = [window bounds];
    if (errors.code != 0 || !isFront(bundle, pid)) return nil;
    if (!windowID || bounds.size.width <= 0 || bounds.size.height <= 0) return emptySnapshot();
    SBObject<SAFrontQSpacePane> *pane = (SBObject<SAFrontQSpacePane> *)[window activatedPane];
    if (errors.code != 0 || !isFront(bundle, pid)) return nil;
    if (!pane) return emptySnapshot();
    NSString *paneID = [pane id];
    if (errors.code != 0 || !paneID || !isFront(bundle, pid)) return nil;
    NSString *url = [(id<SAFrontQSpaceItem>)[pane rootItem] urlstr];
    if (errors.code != 0 && errors.code != -1728) return nil;
    if (!isFront(bundle, pid)) return nil;
    errors.code = 0;
    return @{ @"windows": @[ @{ @"id": windowID, @"front": @YES,
                                @"panes": @[ @{ @"id": paneID, @"active": @YES,
                                                 @"url": url ?: [NSNull null] } ] } ] };
}

static NSDictionary *readRequest(NSDictionary *request) {
    NSNumber *identifier = request[@"id"];
    NSString *source = request[@"source"];
    NSNumber *pidNumber = request[@"pid"];
    if (![identifier isKindOfClass:NSNumber.class] || ![source isKindOfClass:NSString.class] ||
        ![pidNumber isKindOfClass:NSNumber.class] || pidNumber.intValue <= 0) {
        return @{ @"id": identifier ?: @0, @"status": @"invalid" };
    }
    BOOL finder = [source isEqualToString:@"finder"];
    if (!finder && ![source isEqualToString:@"qspace"]) return @{ @"id": identifier, @"status": @"invalid" };
    NSString *bundle = finder ? @"com.apple.finder" : @"com.jinghaoshe.qspace.pro";
    pid_t pid = pidNumber.intValue;
    if (!isFront(bundle, pid)) return @{ @"id": identifier, @"status": @"stale" };
    @try {
        SBApplication *app = [SBApplication applicationWithBundleIdentifier:bundle];
        if (!app) return @{ @"id": identifier, @"status": @"unavailable" };
        SAReadErrors *errors = [SAReadErrors new];
        app.delegate = errors;
        app.timeout = 120; // Apple Event ticks: two seconds per reply.
        NSDictionary *snapshot = finder ? readFinder(app, errors, bundle, pid) : readQSpace(app, errors, bundle, pid);
        if (!isFront(bundle, pid)) return @{ @"id": identifier, @"status": @"stale" };
        if (snapshot && errors.code == 0) return @{ @"id": identifier, @"status": @"ready", @"snapshot": snapshot };
        return @{ @"id": identifier, @"status": errors.code == -1743 ? @"automation_denied" : @"unavailable" };
    } @catch (NSException *exception) {
        return @{ @"id": identifier, @"status": @"unavailable" };
    }
}

int main(void) {
    @autoreleasepool {
        // Each request gets its own autorelease pool; no window or pane state is retained.
        char *line = NULL;
        size_t capacity = 0;
        while (getline(&line, &capacity, stdin) > 0) {
            @autoreleasepool {
                NSData *input = [[NSString stringWithUTF8String:line] dataUsingEncoding:NSUTF8StringEncoding];
                NSDictionary *request = [NSJSONSerialization JSONObjectWithData:input options:0 error:NULL];
                if (![request isKindOfClass:NSDictionary.class]) continue;
                NSDictionary *reply = readRequest(request);
                NSData *output = [NSJSONSerialization dataWithJSONObject:reply options:0 error:NULL];
                if (output) { fwrite(output.bytes, 1, output.length, stdout); fputc('\n', stdout); fflush(stdout); }
            }
        }
        free(line);
    }
    return 0;
}
