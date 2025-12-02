#!/usr/bin/env python3
"""
Quantize Core ML model for faster inference and smaller size.

Options:
  - int8: 1/4 size, best for ANE (Apple Neural Engine)
  - float16: 1/2 size, good balance

Usage:
    python3 scripts/quantize_coreml.py
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

def main():
    """Main quantization interface"""
    print("=" * 70)
    print("CORE ML MODEL QUANTIZATION")
    print("=" * 70)
    print("\nQuantization methods:")
    print("1. int8    - Smallest (~25% original), fastest with ANE, <1% accuracy loss")
    print("2. float16 - Medium (~50% original), good balance")
    print("3. Both")
    print("\nRecommendation: Start with int8 for production (best ANE utilization)")
    
    choice = input("\nEnter choice (1/2/3) [default=1]: ").strip() or "1"
    
    try:
        if choice in ['1', '3']:
            quantize_to_int8()
        
        if choice in ['2', '3']:
            quantize_to_float16()
        
        print("\n" + "=" * 70)
        print("✓ Quantization complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Test latency and accuracy with quantized models on device")
        print("2. Choose best performing variant for production")
        print("3. Update app to use selected model")
        
    except Exception as e:
        print(f"\n✗ Quantization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
