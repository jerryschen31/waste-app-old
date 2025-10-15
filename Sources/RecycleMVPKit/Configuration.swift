import Foundation

/// Central configuration keys for the Recycle MVP core logic.
public enum RecycleConfig {
    /// Minimum confidence score before an item is considered confidently classified.
    public static var defaultConfidenceThreshold: Double { 0.7 }

    /// Timeout (seconds) for inference operations before falling back to manual flow.
    public static var inferenceTimeout: TimeInterval { 1.5 }
}
