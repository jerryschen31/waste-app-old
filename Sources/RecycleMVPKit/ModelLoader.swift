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
            let val = (num as? NSNumber).map { Double(truncating: $0) } ?? 0.0
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
