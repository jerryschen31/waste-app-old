// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RecycleMVP",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "RecycleMVPKit",
            targets: ["RecycleMVPKit"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "RecycleMVPKit",
            path: "Sources/RecycleMVPKit"
        ),
        .testTarget(
            name: "RecycleMVPKitTests",
            dependencies: ["RecycleMVPKit"],
            path: "Tests/RecycleMVPKitTests"
        )
    ]
)
