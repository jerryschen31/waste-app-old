import XCTest
@testable import RecycleMVPKit

final class ConfigurationTests: XCTestCase {
    func testDefaultConfidenceThresholdWithinExpectedRange() {
        XCTAssertGreaterThanOrEqual(RecycleConfig.defaultConfidenceThreshold, 0.5)
        XCTAssertLessThanOrEqual(RecycleConfig.defaultConfidenceThreshold, 0.9)
    }

    func testInferenceTimeoutReasonable() {
        XCTAssertGreaterThan(RecycleConfig.inferenceTimeout, 0.0)
        XCTAssertLessThanOrEqual(RecycleConfig.inferenceTimeout, 2.0)
    }
}
