# ML Model Pipeline: PyTorch → Core ML

## Overview

This document explains the complete journey from downloading a Hugging Face model to having a binary that your iOS app can run using Core ML.

### Architecture of the Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: DOWNLOAD                                                │
│ Hugging Face (PyTorch format) → Local disk                      │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: INSPECT                                                 │
│ Load model → Review architecture, input/output shapes           │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: TRACE / CONVERT                                         │
│ PyTorch → ONNX (intermediate format) → Core ML                  │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: QUANTIZE (Optional but Recommended)                     │
│ Core ML (float32) → Core ML (int8/float16) [smaller, faster]    │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: VALIDATE                                                │
│ Test inference latency, accuracy, ANE acceleration              │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ RESULT: RecyclingNet11.mlpackage                                │
│ Ready to embed in Xcode project                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Detailed Pipeline

### **Step 1: Download Model from Hugging Face** ✅ DONE

**What happens**: Downloads the PyTorch model weights and configuration from Hugging Face.

**What you got**:
```
models/recycling-net-11/
└── models--prithivMLmods--Recycling-Net-11/snapshots/6205d424ed3b7feb16faa4e599f986663a1dbbb7/
    ├── model.safetensors          (354.4 MB) ← Model weights
    ├── config.json                ← Architecture definition
    ├── preprocessor_config.json   ← Image preprocessing settings
    ├── README.md                  ← Model documentation
    └── checkpoint-*/ directories  ← Training checkpoints (can ignore)
```

**Key concept**: 
- `model.safetensors` = weights (safe format for PyTorch models)
- `config.json` = tells us how many layers, hidden dims, etc.
- `preprocessor_config.json` = tells us image input size, normalization

### Cascade note (FoodDetector + RecyclingNet11)

For Phase 1 we adopt a two-stage cascade applied per detected object in an image:

- `FoodDetector` (binary): run first to decide whether the object is food.
    - If food: return `Compost` immediately; skip multi-class inference.
    - If not food: run `RecyclingNet11` to produce 11 logits and aggregate into 5 disposal categories (`Trash`, `Recycle`, `Compost`, `Biological Waste`, `E-waste`).

This reduces average cost by avoiding the heavier multi-class pass on clearly food examples.

---

### **Step 2: Inspect the Model**

**What happens**: Load the model in Python and understand its structure without converting yet.

**Functions to add to your codebase**:

**File**: `scripts/model_inspection.py`

```python
#!/usr/bin/env python3
"""
Inspect the Recycling-Net-11 model architecture and I/O specs.
Helps understand what conversion will need to handle.
"""

import json
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

MODEL_PATH = "./models/recycling-net-11/models--prithivMLmods--Recycling-Net-11/snapshots/6205d424ed3b7feb16faa4e599f986663a1dbbb7"

def inspect_model():
    """Load and inspect model architecture."""
    
    print("=" * 70)
    print("RECYCLING-NET-11 MODEL INSPECTION")
    print("=" * 70)
    
    # Load configuration
    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print("\n1. MODEL ARCHITECTURE")
    print(f"   Architecture Type: {config.get('architectures', ['unknown'])[0]}")
    print(f"   Base Model: {config.get('model_type', 'unknown')}")
    print(f"   Hidden Dimensions: {config.get('hidden_size', 'N/A')}")
    print(f"   Number of Layers: {config.get('num_hidden_layers', 'N/A')}")
    
    # Load image processor
    processor_path = Path(MODEL_PATH) / "preprocessor_config.json"
    with open(processor_path) as f:
        processor_config = json.load(f)
    
    print("\n2. INPUT PREPROCESSING")
    print(f"   Input Image Size: {processor_config.get('size', {}).get('height', 'N/A')}x{processor_config.get('size', {}).get('width', 'N/A')}")
    print(f"   Normalization Mean: {processor_config.get('image_mean', [])}")
    print(f"   Normalization Std: {processor_config.get('image_std', [])}")
    print(f"   Do Resize: {processor_config.get('do_resize', True)}")
    print(f"   Do Normalize: {processor_config.get('do_normalize', True)}")
    
    # Load model and count parameters
    print("\n3. LOADING MODEL (this may take a moment)...")
    try:
        model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Model Size (float32): ~{total_params * 4 / (1024**2):.1f} MB")
        
        # Get number of output classes
        num_labels = config.get('num_labels', 'unknown')
        print(f"\n4. OUTPUT")
        print(f"   Number of Classes: {num_labels}")
        
        # List class labels if available
        if hasattr(model.config, 'id2label'):
            print(f"   Classes:")
            for idx, label in model.config.id2label.items():
                print(f"      {idx}: {label}")
        
        # Test inference shape
        print(f"\n5. TEST INFERENCE SHAPE")
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        
        # Create dummy input
        from PIL import Image
        import numpy as np
        dummy_image = Image.new('RGB', (processor_config['size']['height'], processor_config['size']['width']))
        inputs = processor(dummy_image, return_tensors="pt")
        
        print(f"   Input tensor shape: {inputs['pixel_values'].shape}")
        print(f"   Input tensor dtype: {inputs['pixel_values'].dtype}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   Output shape: {outputs.logits.shape}")
        print(f"   Output dtype: {outputs.logits.dtype}")
        
        print("\n✓ Model inspection complete!")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")

if __name__ == "__main__":
    inspect_model()
```

**Run it**:
```bash
cd /Users/jerry/gh/waste-app
source ml_env/bin/activate
python3 scripts/model_inspection.py
```

**What you'll learn**:
- Exact input image size (probably 384x384 or 224x224)
- Number of output classes (should be 11)
- Model size and parameter count
- Normalization values needed for preprocessing

---

### **Step 3: Convert to Core ML**

**What happens**: 
1. Load PyTorch model
2. Create a sample input
3. Trace the model (record all operations for conversion)
4. Convert to ONNX (intermediate format, easier for coremltools)
5. Convert ONNX to Core ML

**Why ONNX?** It's a format that bridges PyTorch, TensorFlow, and Core ML. Most reliable conversion path.

**Functions to add to your codebase**:

**File**: `scripts/convert_to_coreml.py`

```python
#!/usr/bin/env python3
"""
Convert Recycling-Net-11 from PyTorch → ONNX → Core ML.

This creates a .mlpackage file that can be embedded in your iOS app.
"""

import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import coremltools as ct
from coremltools.models.neural_network import flexible_shape_utils
import onnx
import onnxruntime as ort

MODEL_PATH = "./models/recycling-net-11/models--prithivMLmods--Recycling-Net-11/snapshots/6205d424ed3b7feb16faa4e599f986663a1dbbb7"
OUTPUT_DIR = "./models/coreml"

def pytorch_to_onnx():
    """Step 1: Convert PyTorch → ONNX"""
    
    print("\n" + "=" * 70)
    print("STEP 1: PYTORCH → ONNX CONVERSION")
    print("=" * 70)
    
    onnx_path = Path(OUTPUT_DIR) / "recycling_net_11.onnx"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load model and processor
    print("Loading model from Hugging Face...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    
    # Get input size from processor config
    config_path = Path(MODEL_PATH) / "preprocessor_config.json"
    with open(config_path) as f:
        processor_config = json.load(f)
    
    height = processor_config['size']['height']
    width = processor_config['size']['width']
    
    print(f"Input image size: {height}x{width}")
    
    # Create dummy input matching expected size
    print("Creating dummy input for tracing...")
    dummy_input = torch.randn(1, 3, height, width)  # (batch=1, channels=3, H, W)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,  # Core ML supports up to opset 14
        verbose=False
    )
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model valid. Size: {onnx_path.stat().st_size / (1024**2):.1f} MB")
    
    return str(onnx_path), height, width


### Notes about conversion environment

- During this session we used a dedicated Python 3.11 virtual environment `./coreml_env` to avoid compatibility issues with `coremltools` and ONNX tooling that appeared on Python 3.14.
- Two conversion paths are supported in the repo now:
    - PyTorch -> ONNX -> Core ML (preferred when ONNX path is reliable)
    - Direct PyTorch tracing -> Core ML (fallback when ONNX conversion has API mismatches)

### Measured latency (development machine)

- `FoodDetector` (binary): mean ~1.19 ms (50 runs)
- `RecyclingNet11` (11-way): mean ~4.59 ms (50 runs)
- Combined cascade (naive sum): ~6 ms on the Apple M-series machine used for conversion and benchmarking. Expect higher latencies on iPhone devices; use these as relative baselines.

def onnx_to_coreml(onnx_path: str, height: int, width: int):
    """Step 2: Convert ONNX → Core ML"""
    
    print("\n" + "=" * 70)
    print("STEP 2: ONNX → CORE ML CONVERSION")
    print("=" * 70)
    
    coreml_path = Path(OUTPUT_DIR) / "RecyclingNet11.mlpackage"
    
    print(f"Converting {onnx_path} to Core ML...")
    
    # Convert ONNX to Core ML
    mlmodel = ct.converters.onnx.convert(
        onnx_path,
        minimum_deployment_target=ct.target.iOS16,  # iOS 16+
        compute_units=ct.ComputeUnit.ALL  # Use CPU, GPU, and ANE
    )
    
    # Update model metadata
    mlmodel.author = "Recycle MVP"
    mlmodel.license = "Apache 2.0"
    mlmodel.short_description = "Waste item classification (11 categories)"
    mlmodel.version = "1.0.0"
    
    # Set input/output descriptions
    mlmodel.input_description['pixel_values'] = f"Input image ({height}x{width} RGB)"
    mlmodel.output_description['logits'] = "Classification logits (11 classes)"
    
    # Save Core ML model
    print(f"Saving Core ML model to: {coreml_path}")
    mlmodel.save(str(coreml_path))
    
    print(f"✓ Core ML model saved. Size: {coreml_path.stat().st_size / (1024**2):.1f} MB")
    
    return str(coreml_path)

def main():
    """Main conversion pipeline"""
    
    print("\n" + "=" * 70)
    print("RECYCLING-NET-11 CONVERSION PIPELINE")
    print("PyTorch → ONNX → Core ML")
    print("=" * 70)
    
    try:
        # Step 1: PyTorch to ONNX
        onnx_path, height, width = pytorch_to_onnx()
        
        # Step 2: ONNX to Core ML
        coreml_path = onnx_to_coreml(onnx_path, height, width)
        
        print("\n" + "=" * 70)
        print("✓ CONVERSION COMPLETE")
        print("=" * 70)
        print(f"\nCore ML model ready at: {coreml_path}")
        print("\nNext steps:")
        print("1. Integrate RecyclingNet11.mlpackage into Xcode project")
        print("2. Add to Sources/RecycleMVPKit as model resource")
        print("3. Implement inference wrapper in Inference Engine module")
        print("4. Run latency tests on target device")
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

**Run it**:
```bash
source ml_env/bin/activate
python3 scripts/convert_to_coreml.py
```

**What it does**:
- Loads PyTorch model + processor
- Exports to ONNX (safe intermediate format)
- Converts ONNX to Core ML
- Sets metadata (version, description, etc.)

**Output**: `RecyclingNet11.mlpackage` (~100-350 MB depending on quantization)

---

### **Step 4: Quantize for Size & Speed** (Optional but Recommended)

**What happens**: Reduce model size by converting float32 weights to int8 (1/4 the size) or float16 (1/2 the size).

**Trade-off**: Slightly lower accuracy (typically <1% impact), but much faster inference + smaller app bundle.

**Functions to add**:

**File**: `scripts/quantize_coreml.py`

```python
#!/usr/bin/env python3
"""
Quantize Core ML model for faster inference and smaller size.

Options:
  - int8: 1/4 size, best for ANE (Apple Neural Engine)
  - float16: 1/2 size, good balance
"""

from pathlib import Path
import coremltools as ct

COREML_PATH = "./models/coreml/RecyclingNet11.mlpackage"
OUTPUT_DIR = "./models/coreml"

def quantize_to_int8():
    """Quantize to int8 (smallest, fastest with ANE)"""
    
    print("\n" + "=" * 70)
    print("QUANTIZING TO INT8")
    print("=" * 70)
    
    print(f"Loading Core ML model: {COREML_PATH}")
    model = ct.models.MLModel(COREML_PATH)
    
    print("Quantizing to int8...")
    quantized_model = ct.models.quantization_utils.quantize_weights(
        model,
        nbits=8,
        quantization_mode="linear_symmetric"
    )
    
    output_path = Path(OUTPUT_DIR) / "RecyclingNet11_int8.mlpackage"
    print(f"Saving quantized model: {output_path}")
    quantized_model.save(str(output_path))
    
    # Compare sizes
    orig_size = Path(COREML_PATH).stat().st_size / (1024**2)
    quant_size = output_path.stat().st_size / (1024**2)
    reduction = ((orig_size - quant_size) / orig_size) * 100
    
    print(f"\n✓ Quantization complete")
    print(f"  Original size: {orig_size:.1f} MB")
    print(f"  Quantized size: {quant_size:.1f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    
    return str(output_path)

def quantize_to_float16():
    """Quantize to float16 (medium size, good balance)"""
    
    print("\n" + "=" * 70)
    print("QUANTIZING TO FLOAT16")
    print("=" * 70)
    
    print(f"Loading Core ML model: {COREML_PATH}")
    model = ct.models.MLModel(COREML_PATH)
    
    print("Quantizing to float16...")
    quantized_model = ct.models.quantization_utils.quantize_weights(
        model,
        nbits=16
    )
    
    output_path = Path(OUTPUT_DIR) / "RecyclingNet11_float16.mlpackage"
    print(f"Saving quantized model: {output_path}")
    quantized_model.save(str(output_path))
    
    # Compare sizes
    orig_size = Path(COREML_PATH).stat().st_size / (1024**2)
    quant_size = output_path.stat().st_size / (1024**2)
    reduction = ((orig_size - quant_size) / orig_size) * 100
    
    print(f"\n✓ Quantization complete")
    print(f"  Original size: {orig_size:.1f} MB")
    print(f"  Quantized size: {quant_size:.1f} MB")
    print(f"  Reduction: {reduction:.1f}%")
    
    return str(output_path)

if __name__ == "__main__":
    print("Core ML Model Quantization")
    print("\nChoose quantization method:")
    print("1. int8   - Smallest (~25% original), fastest with ANE, <1% accuracy loss")
    print("2. float16 - Medium (~50% original), good balance")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        quantize_to_int8()
    
    if choice in ['2', '3']:
        quantize_to_float16()
    
    print("\n" + "=" * 70)
    print("Next: Test latency and accuracy with quantized models on device")
    print("=" * 70)
```

**Run it**:
```bash
source ml_env/bin/activate
python3 scripts/quantize_coreml.py
```

---

### **Step 5: Validate and Test**

**What happens**: Load Core ML model in Swift and measure inference latency + accuracy.

**Functions to add to your Swift codebase**:

**File**: `Sources/RecycleMVPKit/ModelLoader.swift`

```swift
import Foundation
import CoreML
import Vision

/// Loads and manages the Core ML model for waste classification.
public class RecyclingNetModelLoader {
    
    public enum LoadError: Error {
        case modelNotFound
        case loadFailed(String)
        case inferenceError(String)
    }
    
    private var model: MLModel?
    private let modelName: String
    
    public init(modelName: String = "RecyclingNet11") {
        self.modelName = modelName
    }
    
    /// Load the Core ML model from the app bundle.
    /// - Throws: LoadError if model file not found or fails to load
    public func load() throws {
        let modelConfig = MLModelConfiguration()
        // Enable ANE (Neural Engine) if available
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
    
    /// Run inference on an image and return class probabilities.
    /// - Parameters:
    ///   - image: UIImage or CVPixelBuffer to classify
    /// - Returns: Dictionary of class label → confidence score
    public func infer(on pixelBuffer: CVPixelBuffer) throws -> [String: Double] {
        guard let model = self.model else {
            throw LoadError.inferenceError("Model not loaded. Call load() first.")
        }
        
        let input = try RecyclingNetInput(pixelBuffer: pixelBuffer)
        let output = try model.prediction(from: input)
        
        // Parse output logits and convert to probabilities
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw LoadError.inferenceError("Invalid output format")
        }
        
        return softmax(logits: logits)
    }
    
    /// Convert logits to softmax probabilities.
    private func softmax(logits: MLMultiArray) -> [String: Double] {
        var results: [String: Double] = [:]
        let classes = WasteCategory.allCases
        
        // Extract raw logits
        var logitsArray: [Double] = []
        for i in 0..<logits.count {
            if let val = logits[i] as NSNumber? {
                logitsArray.append(Double(truncating: val))
            }
        }
        
        // Compute softmax
        let maxLogit = logitsArray.max() ?? 0.0
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

/// Input structure for Core ML model.
private struct RecyclingNetInput {
    let pixelValues: MLMultiArray
    
    init(pixelBuffer: CVPixelBuffer) throws {
        // Assume 384x384 input (from preprocessor_config.json)
        let width = 384
        let height = 384
        
        // Create MLMultiArray with shape [1, 3, height, width]
        guard let pixelArray = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
                                               dataType: .float32) else {
            throw RecyclingNetModelLoader.LoadError.inferenceError("Failed to create input array")
        }
        
        // TODO: Add image preprocessing (resize, normalize) here
        // For now, this is a placeholder
        
        self.pixelValues = pixelArray
    }
}

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
}
```

**File**: `Tests/RecycleMVPKitTests/ModelLoaderTests.swift`

```swift
import XCTest
@testable import RecycleMVPKit

class ModelLoaderTests: XCTestCase {
    
    var modelLoader: RecyclingNetModelLoader!
    
    override func setUp() {
        super.setUp()
        modelLoader = RecyclingNetModelLoader()
    }
    
    func testModelLoads() throws {
        // Should not throw
        try modelLoader.load()
    }
    
    func testInferenceLatency() throws {
        try modelLoader.load()
        
        // Create dummy image
        let dummyImage = createDummyImage(size: CGSize(width: 384, height: 384))
        
        // Measure inference time
        let startTime = Date()
        _ = try modelLoader.infer(on: dummyImage)
        let elapsed = Date().timeIntervalSince(startTime)
        
        print("Inference latency: \(elapsed * 1000)ms")
        
        // Should be < 600ms on target device
        XCTAssertLessThan(elapsed, 1.5, "Inference exceeded timeout")
    }
    
    // Helper function
    private func createDummyImage(size: CGSize) -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault,
                           Int(size.width),
                           Int(size.height),
                           kCVPixelFormatType_32BGRA,
                           nil,
                           &pixelBuffer)
        return pixelBuffer!
    }
}
```

---

## Summary: Dependencies & Libraries

Here's what gets installed and what each does:

| Library | Purpose | Version | Size |
|---------|---------|---------|------|
| **torch** | PyTorch (deep learning framework) | ~2.0+ | ~400MB |
| **transformers** | Hugging Face models library | ~4.30+ | ~50MB |
| **huggingface-hub** | Download models from HF | ~0.18+ | ~5MB |
| **coremltools** | Convert to Core ML format | ~7.0+ | ~20MB |
| **onnx** | Open Neural Network Exchange | ~1.14+ | ~10MB |
| **onnxruntime** | Run ONNX inference | ~1.15+ | ~30MB |
| **Pillow** | Image processing | ~9.0+ | ~10MB |
| **numpy** | Numerical computing | ~1.24+ | ~20MB |

**Total Python environment size**: ~550 MB (one-time setup)

---

## Functions Added to Your Codebase

### Python Scripts
1. **`scripts/model_inspection.py`** – Inspect model architecture
2. **`scripts/convert_to_coreml.py`** – Main conversion pipeline (PyTorch → ONNX → Core ML)
3. **`scripts/quantize_coreml.py`** – Reduce model size via quantization (optional)

### Swift Classes
1. **`Sources/RecycleMVPKit/ModelLoader.swift`** – Load and run Core ML model
2. **`Tests/RecycleMVPKitTests/ModelLoaderTests.swift`** – Test inference latency

---

## Full Command Sequence

```bash
# 1. Setup (one-time)
cd /Users/jerry/gh/waste-app
python3 -m venv ml_env
source ml_env/bin/activate
pip install torch transformers huggingface-hub coremltools onnx onnxruntime Pillow

# 2. Inspect model
python3 scripts/model_inspection.py

# 3. Convert to Core ML
python3 scripts/convert_to_coreml.py
# Output: ./models/coreml/RecyclingNet11.mlpackage

# 4. (Optional) Quantize
python3 scripts/quantize_coreml.py
# Output: ./models/coreml/RecyclingNet11_int8.mlpackage

# 5. Add to Xcode project
# Copy RecyclingNet11.mlpackage into Xcode, add to RecycleMVPKit target

# 6. Test in Swift
swift test
```

---

## What Happens Inside Each Step (Technical Deep Dive)

### PyTorch Model Format
- **What it is**: A graph of neural network operations + learned weights
- **File format**: `.safetensors` (safer alternative to `.pt`)
- **Size**: Full precision (float32) = ~354MB for Recycling-Net-11

### ONNX (Open Neural Network Exchange)
- **What it is**: Standardized, framework-agnostic model format
- **Why needed**: Bridge between PyTorch and Core ML
- **What happens**: Operations are translated from PyTorch dialect to ONNX dialect
- **Size**: Similar to PyTorch (~354MB)

### Core ML Format
- **What it is**: Apple's native ML model format (`.mlpackage` or `.mlmodelc`)
- **Key features**: Optimizes for ANE (Apple Neural Engine), CPU, GPU
- **Conversion process**: ONNX operations → Core ML operations (some may be fused/reordered for efficiency)
- **Size**: Can be 100-350MB depending on quantization

### Quantization
- **float32** (baseline): 4 bytes per weight = 354MB
- **float16**: 2 bytes per weight = 177MB (50% savings)
- **int8**: 1 byte per weight = 88MB (75% savings)
- **Trade-off**: int8 slightly reduces accuracy (<1% typically) but enables ANE optimization

---

## What Can Go Wrong & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "ModuleNotFoundError" | Missing Python package | `pip install <package>` |
| "opset version not supported" | Core ML doesn't support ONNX opset | Lower opset_version in export (use 14 max) |
| Model too large (>500MB) | No quantization applied | Run quantization script (int8 recommended) |
| "Unsupported operation" | ONNX op not supported by Core ML | May need model retraining or custom conversion |
| Low inference latency on device | Model not using ANE | Ensure `computeUnits = .all` in MLModelConfiguration |

---

## Next: Integrate into iOS App

Once you have `RecyclingNet11.mlpackage`:

1. **Add to Xcode**: Drag into project, ensure embedded in RecycleMVPKit target
2. **Implement `ModelLoader.swift`** (provided above)
3. **Create image preprocessing module** to handle:
   - Resize to 384x384
   - Normalize per model config (mean/std)
   - Convert CVPixelBuffer → MLMultiArray
4. **Wire into Inference Engine** (planned Phase 1 module)
5. **Test on device** with Xcode Instruments (Energy Impact, Network, etc.)

---

## Questions to Resolve Before Proceeding

1. **Model size budget**: Is 88-177MB acceptable for app bundle? (Can be reduced by excluding quantized variant)
2. **Accuracy baseline**: Will you gather a test set to validate 90% accuracy requirement?
3. **ANE prioritization**: Is ANE acceleration a hard requirement, or is CPU fallback acceptable?
4. **Update mechanism**: Will you bundle model in app or download updates over network?

