// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "SystemAudioHelper",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "SystemAudioHelper",
            path: "Sources",
            linkerSettings: [
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("CoreMedia"),
            ]
        ),
    ]
)
