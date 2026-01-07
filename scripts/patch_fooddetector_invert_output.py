#!/usr/bin/env python3
"""
Patch FoodDetector.mlpackage to invert its output, so that food images yield high values and not-food images yield low values.
This script creates a new model: FoodDetector_inverted.mlpackage
"""
import coremltools as ct
from pathlib import Path
import shutil
import tempfile
import os
import numpy as np

# Paths
orig_path = Path("models/coreml/FoodDetector.mlpackage")
out_path = Path("models/coreml/FoodDetector_inverted.mlpackage")

print(f"Loading original model: {orig_path}")
model = ct.models.MLModel(str(orig_path))
spec = model.get_spec()

# Check output name
output_name = spec.description.output[0].name
print(f"Original output name: {output_name}")

# Use coremltools' neural network wrapper to invert the output
from coremltools.models.neural_network import NeuralNetworkBuilder

# The output is a single float (1,1) array, so we can wrap it
input_features = [(output_name, ct.models.datatypes.Array(1,1))]
output_features = [("food_prob", ct.models.datatypes.Array(1,1))]

builder = NeuralNetworkBuilder(input_features, output_features)

# Add layer: temp_neg = -1 * input
builder.add_activation(
    name="negate",
    non_linearity="LINEAR",
    input_name=output_name,
    output_name="temp_neg",
    params=[-1.0, 0.0]
)
# Add layer: food_prob = temp_neg + 1
builder.add_bias(
    name="add_one",
    input_name="temp_neg",
    output_name="food_prob",
    b=np.array([1.0], dtype=np.float32),
    shape_bias=[1]
)

# Save the wrapper model to a temp file
with tempfile.TemporaryDirectory() as tmpdir:
    wrapper_path = Path(tmpdir) / "invert_wrapper.mlmodel"
    ct.models.utils.save_spec(builder.spec, str(wrapper_path))
    print(f"Wrapper model saved: {wrapper_path}")

    # Now pipeline: original model -> wrapper
    pipeline = ct.models.pipeline.Pipeline(
        input_features=[(spec.description.input[0].name, ct.models.datatypes.Array(*spec.description.input[0].type.multiArrayType.shape))],
        output_features=[("food_prob", ct.models.datatypes.Array(1,1))]
    )
    pipeline.add_model(model)
    wrapper_model = ct.models.MLModel(str(wrapper_path))
    pipeline.add_model(wrapper_model)
    pipeline_spec = pipeline.spec
    # Set input/output descriptions directly
    del pipeline_spec.description.input[:]
    pipeline_spec.description.input.extend(spec.description.input)
    # Construct output description
    from coremltools.proto import Model_pb2
    output_desc = Model_pb2.FeatureDescription()
    output_desc.name = "food_prob"
    output_desc.type.CopyFrom(builder.spec.description.output[0].type)
    output_desc.shortDescription = "Probability of food (inverted, correct polarity)"
    del pipeline_spec.description.output[:]
    pipeline_spec.description.output.extend([output_desc])
    # Save the new pipeline model
    print(f"Saving inverted model to: {out_path}")
    ct.models.utils.save_spec(pipeline_spec, str(out_path))

print("Done. New model with correct output polarity saved as FoodDetector_inverted.mlpackage.")
