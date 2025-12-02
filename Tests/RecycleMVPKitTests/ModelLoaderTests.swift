import XCTest
@testable import RecycleMVPKit

class ModelLoaderTests: XCTestCase {
    
    var modelLoader: RecyclingNetModelLoader!
    
    override func setUp() {
        super.setUp()
        modelLoader = RecyclingNetModelLoader()
    }
    
    override func tearDown() {
        modelLoader = nil
        super.tearDown()
    }
    
    // MARK: - Model Loading Tests
    
    func testModelLoads() throws {
        // Should load without throwing
        try modelLoader.load()
    }
    
    func testModelFailsIfNotFound() throws {
        let badLoader = RecyclingNetModelLoader(modelName: "NonExistentModel")
        XCTAssertThrowsError(try badLoader.load()) { error in
            if case RecyclingNetModelLoader.LoadError.modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        }
    }
    
    // MARK: - Inference Tests
    
    func testInferenceFailsWithoutLoading() throws {
        let dummyBuffer = createDummyPixelBuffer(width: 384, height: 384)
        
        XCTAssertThrowsError(try modelLoader.infer(on: dummyBuffer)) { error in
            if case RecyclingNetModelLoader.LoadError.inferenceError = error {
                // Expected - model not loaded
            } else {
                XCTFail("Expected inferenceError, got: \(error)")
            }
        }
    }
    
    func testInferenceProducesValidOutput() throws {
        try modelLoader.load()
        
        let dummyBuffer = createDummyPixelBuffer(width: 384, height: 384)
        let result = try modelLoader.infer(on: dummyBuffer)
        
        // Should return all 11 categories
        XCTAssertEqual(result.count, WasteCategory.allCases.count)
        
        // All probabilities should be between 0 and 1
        for (_, prob) in result {
            XCTAssertGreaterThanOrEqual(prob, 0.0)
            XCTAssertLessThanOrEqual(prob, 1.0)
        }
        
        // Probabilities should sum to approximately 1.0
        let sum = result.values.reduce(0.0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.01)
    }
    
    // MARK: - Performance Tests
    
    func testInferenceLatency() throws {
        try modelLoader.load()
        
        let dummyBuffer = createDummyPixelBuffer(width: 384, height: 384)
        
        // Measure inference time
        let startTime = CFAbsoluteTimeGetCurrent()
        _ = try modelLoader.infer(on: dummyBuffer)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        let elapsedMs = elapsed * 1000
        print(String(format: "Inference latency: %.1f ms", elapsedMs))
        
        // Must be under 1500ms per requirements (worst case)
        // Median should be <600ms on target device
        XCTAssertLessThan(elapsed, 1.5, "Inference exceeded 1500ms timeout")
    }
    
    func testInferenceLatencyMultiple() throws {
        try modelLoader.load()
        
        let dummyBuffer = createDummyPixelBuffer(width: 384, height: 384)
        var latencies: [Double] = []
        
        // Run 5 times and measure
        for _ in 0..<5 {
            let startTime = CFAbsoluteTimeGetCurrent()
            _ = try modelLoader.infer(on: dummyBuffer)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            latencies.append(elapsed * 1000)
        }
        
        let avgLatency = latencies.reduce(0, +) / Double(latencies.count)
        let maxLatency = latencies.max() ?? 0
        let minLatency = latencies.min() ?? 0
        
        print(String(format: "Avg latency: %.1f ms | Min: %.1f ms | Max: %.1f ms", avgLatency, minLatency, maxLatency))
        
        // Average should be reasonable
        XCTAssertLessThan(avgLatency, 1000, "Average inference latency too high")
    }
    
    // MARK: - Helper Methods
    
    /// Create a dummy BGRA pixel buffer of specified size
    private func createDummyPixelBuffer(width: Int, height: Int) -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            fatalError("Failed to create pixel buffer")
        }
        
        // Fill with dummy data
        CVPixelBufferLockBaseAddress(buffer, .readAndWrite)
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let dataSize = CVPixelBufferGetDataSize(buffer)
        memset(pixelData, 128, dataSize)  // Mid-gray
        CVPixelBufferUnlockBaseAddress(buffer, .readAndWrite)
        
        return buffer
    }
}

// MARK: - Waste Category Tests

class WasteCategoryTests: XCTestCase {
    
    func testAllCategoriesExist() {
        let categories = WasteCategory.allCases
        XCTAssertEqual(categories.count, 11, "Should have exactly 11 waste categories")
    }
    
    func testCategoryFromIndex() {
        XCTAssertEqual(WasteCategory.fromIndex(0), .paper)
        XCTAssertEqual(WasteCategory.fromIndex(10), .other)
        XCTAssertNil(WasteCategory.fromIndex(11))
        XCTAssertNil(WasteCategory.fromIndex(-1))
    }
    
    func testCategoryRawValues() {
        XCTAssertEqual(WasteCategory.paper.rawValue, "Paper")
        XCTAssertEqual(WasteCategory.plastic.rawValue, "Plastic")
        XCTAssertEqual(WasteCategory.glass.rawValue, "Glass")
    }
}
