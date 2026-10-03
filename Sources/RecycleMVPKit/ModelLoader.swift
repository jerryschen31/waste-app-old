import Foundation
import CoreML
import Vision
import CoreImage

/// Loads and manages the Core ML model for waste classification.
public class RecyclingNetModelLoader {
    
    public enum LoadError: Error {
        case modelNotFound
        case loadFailed(String)
        case inferenceError(String)
        case preprocessingError(String)
    }
    
    private var model: MLModel?
    private let modelName: String
    
    /// Initialize the model loader.
    /// - Parameter modelName: Name of the .mlmodelc file (without extension)
    public init(modelName: String = "RecyclingNet11") {
        self.modelName = modelName
    }
    
    /// Load the Core ML model from the app bundle.
    /// - Throws: LoadError if model file not found or fails to load
    public func load() throws {
        let modelConfig = MLModelConfiguration()
        // Enable ANE (Neural Engine) if available, fallback to CPU/GPU
        modelConfig.computeUnits = .all
        
        guard let modelURL = Bundle.main.url(
            forResource: modelName,
            withExtension: "mlmodelc"
        ) else {
            throw LoadError.modelNotFound
        }
        
        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: modelConfig)
        } catch {
            throw LoadError.loadFailed("Failed to load Core ML model: \(error)")
        }
    }
    
    /// Run inference on a CVPixelBuffer and return class probabilities.
    /// - Parameters:
    ///   - pixelBuffer: Image pixel buffer to classify (384x384 RGB expected)
    /// - Returns: Dictionary of class label → confidence score (0.0-1.0)
    /// - Throws: LoadError if model not loaded or inference fails
    public func infer(on pixelBuffer: CVPixelBuffer) throws -> [String: Double] {
        guard let model = self.model else {
            throw LoadError.inferenceError("Model not loaded. Call load() first.")
        }
        
        // Preprocess image
        let preprocessed = try preprocessImage(pixelBuffer)
        
        // Create input
        let input = try RecyclingNetInput(pixelArray: preprocessed)
        
        // Run inference
        let output = try model.prediction(from: input)
        
        // Parse output logits and convert to probabilities (per-class)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw LoadError.inferenceError("Invalid output format from model")
        }

        let classProbs = classProbabilities(from: logits)
        // Convert to [String: Double] for compatibility with existing callers
        var results: [String: Double] = [:]
        for (cls, prob) in classProbs {
            results[cls.rawValue] = prob
        }

        return results
    }
    
    /// Preprocess image to match model input requirements (384x384, normalized).
    /// Resizes to 384×384 and normalizes with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5].
    /// - Parameter pixelBuffer: Raw camera frame (any size, any orientation)
    /// - Returns: Preprocessed pixel buffer (384x384 RGB)
    /// - Throws: LoadError if image processing fails
    private func preprocessImage(_ pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        let targetWidth = 384
        let targetHeight = 384
        
        // Step 1: Create a color space and image context for resizing
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        // Convert CVPixelBuffer to CGImage for processing
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(
            ciImage,
            from: ciImage.extent
        ) else {
            throw LoadError.preprocessingError("Failed to create CGImage from CVPixelBuffer")
        }
        
        // Step 2: Resize image to 384x384
        guard let resizedImage = resizeImage(cgImage, to: CGSize(width: targetWidth, height: targetHeight)) else {
            throw LoadError.preprocessingError("Failed to resize image to 384x384")
        }
        
        // Step 3: Create output CVPixelBuffer for the resized, normalized image
        var outputBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetWidth,
            targetHeight,
            kCVPixelFormatType_32BGRA,
            nil,
            &outputBuffer
        )
        
        guard status == kCVReturnSuccess, let output = outputBuffer else {
            throw LoadError.preprocessingError("Failed to create output CVPixelBuffer")
        }
        
        // Step 4: Draw resized image into output buffer with normalization
        CVPixelBufferLockBaseAddress(output, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(output, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let drawContext = CGContext(
            data: CVPixelBufferGetBaseAddress(output),
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(output),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw LoadError.preprocessingError("Failed to create drawing context")
        }
        
        // Draw the resized image
        drawContext.draw(resizedImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        
        // Step 5: Normalize pixel values
        // Normalize: (pixel / 255.0 - mean) / std
        // With mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
        normalizePixelBuffer(output, mean: 0.5, std: 0.5)
        
        return output
    }
    
    /// Resize a CGImage to the specified size.
    private func resizeImage(_ cgImage: CGImage, to size: CGSize) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return context.makeImage()
    }
    
    /// Normalize pixel values in-place: (pixel / 255.0 - mean) / std
    private func normalizePixelBuffer(_ pixelBuffer: CVPixelBuffer, mean: Float, std: Float) {
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        // Treat as raw bytes (BGRA 8-bit format)
        let pixelData = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4  // 4 bytes per pixel (BGRA)
                
                // Note: BGRA order, so normalize B, G, R (skip A)
                for c in 0..<3 {  // B, G, R channels
                    let byteValue = Float(pixelData[offset + c])
                    // Normalize: (pixel / 255.0 - mean) / std, then scale back to 0-255
                    let normalized = ((byteValue / 255.0) - mean) / std
                    // Clamp to valid range and convert back to UInt8
                    let clamped = max(0, min(255, Int(normalized * 255.0)))
                    pixelData[offset + c] = UInt8(clamped)
                }
                // Leave alpha channel untouched
            }
        }
    }
    
    private func softmaxArray(_ logitsArray: [Double]) -> [Double] {
        let maxLogit = logitsArray.max() ?? 0.0
        let exps = logitsArray.map { exp($0 - maxLogit) }
        let sumExp = exps.reduce(0.0, +)
        guard sumExp > 0 else { return exps.map { _ in 0.0 } }
        return exps.map { $0 / sumExp }
    }

    /// Convert raw logits to per-class probabilities keyed by `WasteCategory`.
    private func classProbabilities(from logits: MLMultiArray) -> [WasteCategory: Double] {
        var logitsArray: [Double] = []
        for i in 0..<logits.count {
            let num = logits[i]
            let val = Double(truncating: num as NSNumber)
            logitsArray.append(val)
        }

        let probs = softmaxArray(logitsArray)
        var result: [WasteCategory: Double] = [:]
        for (idx, p) in probs.enumerated() {
            if let wc = WasteCategory.fromIndex(idx) {
                result[wc] = p
            }
        }
        return result
    }

    /// Aggregate per-class probabilities into the reduced 5-category disposal mapping.
    /// - Parameter logits: Raw output from model
    /// - Returns: Mapping of aggregated disposal category name -> probability
    public func inferAggregated(on pixelBuffer: CVPixelBuffer) throws -> [String: Double] {
        guard let model = self.model else {
            throw LoadError.inferenceError("Model not loaded. Call load() first.")
        }

        let preprocessed = try preprocessImage(pixelBuffer)
        let input = try RecyclingNetInput(pixelArray: preprocessed)
        let output = try model.prediction(from: input)

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw LoadError.inferenceError("Invalid output format from model")
        }

        let classProbs = classProbabilities(from: logits)

        // Aggregation mapping from fine-grained model classes to the requested 5 categories
        var aggregated: [DisposalCategory: Double] = [:]
        for (cls, prob) in classProbs {
            let agg = cls.toDisposalCategory()
            aggregated[agg, default: 0.0] += prob
        }

        // Convert to [String: Double]
        var results: [String: Double] = [:]
        for (k, v) in aggregated {
            results[k.rawValue] = v
        }

        return results
    }
}

/// Input structure for Core ML model - handles tensor creation
private class RecyclingNetInput: MLFeatureProvider {
    let pixelValues: MLMultiArray
    
    var featureNames: Set<String> {
        return ["pixel_values"]
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "pixel_values" else { return nil }
        return MLFeatureValue(multiArray: pixelValues)
    }
    
    /// Create input from preprocessed pixel buffer.
    /// - Parameter pixelArray: 384x384 RGB pixel buffer (normalized)
    init(pixelArray: CVPixelBuffer) throws {
        // Model expects: [1, 3, 384, 384] float32
        let height = 384
        let width = 384
        let channels = 3
        
        guard let multiArray = try? MLMultiArray(
            shape: [NSNumber(value: 1), NSNumber(value: channels), NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        ) else {
            throw RecyclingNetModelLoader.LoadError.inferenceError("Failed to create input tensor")
        }
        
        // Extract pixel data from CVPixelBuffer and convert to tensor
        CVPixelBufferLockBaseAddress(pixelArray, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(pixelArray, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelArray) else {
            throw RecyclingNetModelLoader.LoadError.preprocessingError("Unable to access pixel buffer data")
        }
        
        let pixelData = baseAddress.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelArray)
        
        // Copy and convert pixel data to tensor in CHW format (channels × height × width)
        // CVPixelBuffer is in BGRA format, so extract B, G, R channels
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * 4
                
                // Extract normalized pixel values (0.0-1.0 after preprocessing normalization)
                let b = Float(pixelData[pixelOffset + 0]) / 255.0
                let g = Float(pixelData[pixelOffset + 1]) / 255.0
                let r = Float(pixelData[pixelOffset + 2]) / 255.0
                
                // Write to tensor in CHW order (channels first)
                let rIdx = 0 * height * width + y * width + x
                let gIdx = 1 * height * width + y * width + x
                let bIdx = 2 * height * width + y * width + x
                
                multiArray[rIdx] = NSNumber(value: r)
                multiArray[gIdx] = NSNumber(value: g)
                multiArray[bIdx] = NSNumber(value: b)
            }
        }
        
        self.pixelValues = multiArray
    }
}

/// Waste categories that the model can classify.
/// These should match model's id2label mapping.
public enum WasteCategory: String, CaseIterable {
    // Exact labels returned by the model (index order must match model output)
    case aluminium = "aluminium"
    case batteries = "batteries"
    case cardboard = "cardboard"
    case disposablePlates = "disposable plates"
    case glass = "glass"
    case hardPlastic = "hard plastic"
    case paper = "paper"
    case paperTowel = "paper towel"
    case polystyrene = "polystyrene"
    case softPlastics = "soft plastics"
    case takeawayCups = "takeaway cups"

    /// Get category from class index (0-10). The order MUST match the model's output order.
    public static func fromIndex(_ index: Int) -> WasteCategory? {
        guard index >= 0 && index < allCases.count else { return nil }
        return allCases[index]
    }
}

/// High-level disposal categories the app will surface (reduced set).
public enum DisposalCategory: String {
    case trash = "Trash"
    case compost = "Compost"
    case recycle = "Recycle"
    case ewaste = "E-waste"
    case biological = "Biological Waste"
}

extension WasteCategory {
    /// Map fine-grained model classes to the reduced disposal categories.
    /// Conservative defaults chosen per your guidance.
    func toDisposalCategory() -> DisposalCategory {
        switch self {
        case .aluminium, .cardboard, .glass, .hardPlastic, .paper, .softPlastics:
            // Core recyclable materials
            return .recycle
        case .batteries:
            return .ewaste
        case .paperTowel, .disposablePlates, .takeawayCups:
            // Paper towel and some single-use paper items -> compost
            return .compost
        case .polystyrene:
            // User requested polystyrene -> Trash
            return .trash
        }
    }
}

/// Loads and manages the FoodDetector Core ML model.
/// Note: The FoodDetector model outputs inverted probabilities (high for non-food, low for food).
/// This loader automatically inverts the output so that 1.0 = food, 0.0 = not-food.
public class FoodDetectorModelLoader {
    
    public enum LoadError: Error {
        case modelNotFound
        case loadFailed(String)
        case inferenceError(String)
        case preprocessingError(String)
    }
    
    private var model: MLModel?
    private let modelName: String
    
    /// Initialize the FoodDetector model loader.
    /// - Parameter modelName: Name of the .mlmodelc file (without extension)
    public init(modelName: String = "FoodDetector") {
        self.modelName = modelName
    }
    
    /// Load the Core ML model from the app bundle.
    /// - Throws: LoadError if model file not found or fails to load
    public func load() throws {
        let modelConfig = MLModelConfiguration()
        // Enable ANE (Neural Engine) if available, fallback to CPU/GPU
        modelConfig.computeUnits = .all
        
        guard let modelURL = Bundle.main.url(
            forResource: modelName,
            withExtension: "mlmodelc"
        ) else {
            throw LoadError.modelNotFound
        }
        
        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: modelConfig)
        } catch {
            throw LoadError.loadFailed("Failed to load FoodDetector Core ML model: \(error)")
        }
    }
    
    /// Run inference on a CVPixelBuffer and return food probability.
    /// The model outputs are automatically inverted so that 1.0 = food, 0.0 = not-food.
    /// - Parameters:
    ///   - pixelBuffer: Image pixel buffer to classify (224x224 RGB expected)
    /// - Returns: Food probability (0.0-1.0, where 1.0 = food)
    /// - Throws: LoadError if model not loaded or inference fails
    public func infer(on pixelBuffer: CVPixelBuffer) throws -> Double {
        guard let model = self.model else {
            throw LoadError.inferenceError("Model not loaded. Call load() first.")
        }
        
        // Preprocess image
        let preprocessed = try preprocessImage(pixelBuffer)
        
        // Create input
        let input = try FoodDetectorInput(pixelArray: preprocessed)
        
        // Run inference
        let output = try model.prediction(from: input)
        
        // Extract raw output (food_prob or var_1626 depending on model version)
        var rawProb: Double = 0.0
        if let foodProb = output.featureValue(for: "food_prob")?.multiArrayValue {
            rawProb = Double(truncating: foodProb[0] as NSNumber)
        } else if let var1626 = output.featureValue(for: "var_1626")?.multiArrayValue {
            rawProb = Double(truncating: var1626[0] as NSNumber)
        } else {
            throw LoadError.inferenceError("Invalid output format from FoodDetector")
        }
        
        // IMPORTANT: FoodDetector outputs are inverted
        // (high values for not-food, low values for food)
        // So we invert: food_prob = 1.0 - raw_prob
        let foodProb = 1.0 - rawProb
        
        return foodProb
    }
    
    /// Preprocess image to match FoodDetector input requirements (224x224, no normalization).
    /// FoodDetector expects raw 0-255 pixel values, not normalized.
    /// - Parameter pixelBuffer: Raw camera frame (any size, any orientation)
    /// - Returns: Preprocessed pixel buffer (224x224 RGB, 0-255)
    /// - Throws: LoadError if image processing fails
    private func preprocessImage(_ pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        let targetWidth = 224
        let targetHeight = 224
        
        // Step 1: Create a color space and image context for resizing
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        // Convert CVPixelBuffer to CGImage for processing
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(
            ciImage,
            from: ciImage.extent
        ) else {
            throw LoadError.preprocessingError("Failed to create CGImage from CVPixelBuffer")
        }
        
        // Step 2: Resize image to 224x224
        guard let resizedImage = resizeImage(cgImage, to: CGSize(width: targetWidth, height: targetHeight)) else {
            throw LoadError.preprocessingError("Failed to resize image to 224x224")
        }
        
        // Step 3: Create output CVPixelBuffer for the resized image
        var outputBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetWidth,
            targetHeight,
            kCVPixelFormatType_32BGRA,
            nil,
            &outputBuffer
        )
        
        guard status == kCVReturnSuccess, let output = outputBuffer else {
            throw LoadError.preprocessingError("Failed to create output CVPixelBuffer")
        }
        
        // Step 4: Draw resized image into output buffer (NO normalization for FoodDetector)
        CVPixelBufferLockBaseAddress(output, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(output, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let drawContext = CGContext(
            data: CVPixelBufferGetBaseAddress(output),
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(output),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw LoadError.preprocessingError("Failed to create drawing context")
        }
        
        drawContext.draw(resizedImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        
        return output
    }
    
    /// Resize a CGImage to the specified size.
    private func resizeImage(_ cgImage: CGImage, to size: CGSize) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return context.makeImage()
    }
}

/// Input structure for FoodDetector Core ML model
private class FoodDetectorInput: MLFeatureProvider {
    let input: MLMultiArray
    
    var featureNames: Set<String> {
        return ["input"]
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "input" else { return nil }
        return MLFeatureValue(multiArray: input)
    }
    
    /// Create input from preprocessed pixel buffer.
    /// - Parameter pixelArray: 224x224 RGB pixel buffer (raw 0-255, not normalized)
    init(pixelArray: CVPixelBuffer) throws {
        // Model expects: [1, 3, 224, 224] float32
        let height = 224
        let width = 224
        let channels = 3
        
        guard let multiArray = try? MLMultiArray(
            shape: [NSNumber(value: 1), NSNumber(value: channels), NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        ) else {
            throw FoodDetectorModelLoader.LoadError.inferenceError("Failed to create input tensor")
        }
        
        // Extract pixel data from CVPixelBuffer and convert to tensor
        CVPixelBufferLockBaseAddress(pixelArray, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(pixelArray, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelArray) else {
            throw FoodDetectorModelLoader.LoadError.preprocessingError("Unable to access pixel buffer data")
        }
        
        let pixelData = baseAddress.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelArray)
        
        // Copy pixel data to tensor in CHW format (channels × height × width)
        // CVPixelBuffer is in BGRA format, extract B, G, R channels (raw 0-255, no normalization)
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * 4
                
                // Extract raw pixel values (0-255)
                let b = Float(pixelData[pixelOffset + 0])
                let g = Float(pixelData[pixelOffset + 1])
                let r = Float(pixelData[pixelOffset + 2])
                
                // Write to tensor in CHW order (channels first)
                let rIdx = 0 * height * width + y * width + x
                let gIdx = 1 * height * width + y * width + x
                let bIdx = 2 * height * width + y * width + x
                
                multiArray[rIdx] = NSNumber(value: r)
                multiArray[gIdx] = NSNumber(value: g)
                multiArray[bIdx] = NSNumber(value: b)
            }
        }
        
        self.input = multiArray
    }
}
