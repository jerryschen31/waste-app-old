import Foundation
import CoreML
import Vision

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
        
        // Parse output logits and convert to probabilities
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw LoadError.inferenceError("Invalid output format from model")
        }
        
        return softmax(logits: logits)
    }
    
    /// Preprocess image to match model input requirements (384x384, normalized).
    /// - Parameter pixelBuffer: Raw camera frame
    /// - Returns: Preprocessed pixel buffer (384x384)
    private func preprocessImage(_ pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        // TODO: Implement full preprocessing:
        // 1. Resize to 384x384
        // 2. Normalize: (pixel - mean) / std
        //    mean = [0.485, 0.456, 0.406]
        //    std = [0.229, 0.224, 0.225]
        // 3. Convert BGR to RGB if needed
        
        // For now, return as-is (assumes input is already 384x384)
        // This is a placeholder - full implementation needed
        return pixelBuffer
    }
    
    /// Convert raw logits to softmax probabilities.
    /// - Parameter logits: Raw output from model (11 values)
    /// - Returns: Dictionary mapping waste category to probability
    private func softmax(logits: MLMultiArray) -> [String: Double] {
        var results: [String: Double] = [:]
        let classes = WasteCategory.allCases
        
        // Extract raw logits as doubles
        var logitsArray: [Double] = []
        for i in 0..<logits.count {
            if let val = logits[i] as? NSNumber {
                logitsArray.append(Double(truncating: val))
            }
        }
        
        // Compute softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
        let maxLogit = logitsArray.max() ?? 0.0  // Subtract max for numerical stability
        let exps = logitsArray.map { exp($0 - maxLogit) }
        let sumExp = exps.reduce(0.0, +)
        let probabilities = exps.map { $0 / sumExp }
        
        // Map to class labels
        for (idx, prob) in probabilities.enumerated() {
            if idx < classes.count {
                results[classes[idx].rawValue] = prob
            }
        }
        
        return results
    }
}

/// Input structure for Core ML model - handles tensor creation
private struct RecyclingNetInput {
    let pixelValues: MLMultiArray
    
    /// Create input from preprocessed pixel buffer.
    /// - Parameter pixelArray: 384x384 RGB pixel buffer
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
        
        // TODO: Copy pixel buffer data into multiArray with normalization
        // Current placeholder - full implementation needed
        
        self.pixelValues = multiArray
    }
}

/// Waste categories that the model can classify.
/// These should match model's id2label mapping.
public enum WasteCategory: String, CaseIterable {
    case paper = "Paper"
    case cardboard = "Cardboard"
    case biological = "Biological"
    case metals = "Metals"
    case plastic = "Plastic"
    case glass = "Glass"
    case clothes = "Clothes"
    case shoes = "Shoes"
    case battery = "Battery"
    case trash = "Trash"
    case other = "Other"
    
    /// Get category from class index (0-10)
    public static func fromIndex(_ index: Int) -> WasteCategory? {
        guard index >= 0 && index < allCases.count else { return nil }
        return allCases[index]
    }
}
