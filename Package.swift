// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SmartApp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "SmartApp", targets: ["SmartApp"]),
        .executable(name: "smartapp-cli", targets: ["SmartAppCLI"]),
        .executable(name: "smartapp-browser-host", targets: ["BrowserNativeHost"]),
        .executable(name: "smartapp-directory-host", targets: ["DirectoryNativeHost"]),
        .library(name: "IntentCore", targets: ["IntentCore"])
    ],
    targets: [
        .systemLibrary(name: "CSQLite"),
        .target(name: "IntentCore", dependencies: ["CSQLite"]),
        .target(name: "FileRecallCore"),
        .target(name: "FileRecallAX", dependencies: ["FileRecallCore"],
                linkerSettings: [.linkedFramework("ApplicationServices"), .linkedFramework("AppKit")]),
        .executableTarget(name: "SmartApp", dependencies: ["IntentCore", "FileRecallCore", "FileRecallAX"],
                          linkerSettings: [.linkedFramework("Carbon")]),
        .executableTarget(name: "SmartAppCLI", dependencies: ["IntentCore"]),
        .executableTarget(name: "BrowserNativeHost", dependencies: ["IntentCore"]),
        .executableTarget(name: "DirectoryNativeHost", path: "Sources/DirectoryNativeHost",
                          linkerSettings: [.linkedFramework("AppKit"), .linkedFramework("ScriptingBridge")]),
        .testTarget(name: "IntentCoreTests", dependencies: ["IntentCore"]),
        .testTarget(name: "SmartAppTests", dependencies: ["SmartApp", "FileRecallCore"])
    ]
)
