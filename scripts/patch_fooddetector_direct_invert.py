#!/usr/bin/env python3
"""
Directly patch FoodDetector.mlpackage to add an output inversion layer (food_prob = 1 - var_1626).
Creates FoodDetector_fixed.mlmodel (single model, portable, no pipeline).
"""
import coremltools as ct
from pathlib import Path
import numpy as np
from coremltools.models.neural_network import NeuralNetworkBuilder

orig_path = Path("models/coreml/FoodDetector.mlpackage")
out_path = Path("models/coreml/FoodDetector_fixed.mlmodel")

print(f"Loading original model: {orig_path}")
model = ct.models.MLModel(str(orig_path))
spec = model.get_spec()

# Check output name
output_name = spec.description.output[0].name
print(f"Original output name: {output_name}")

# Only works for neuralNetwork or neuralNetworkClassifier
if not (spec.HasField('neuralNetwork') or spec.HasField('neuralNetworkClassifier')):
    raise RuntimeError("Model is not a neural network. Direct patch not supported.")

# Add a new layer to invert the output
if spec.HasField('neuralNetwork'):
    layers = spec.neuralNetwork.layers
elif spec.HasField('neuralNetworkClassifier'):
    layers = spec.neuralNetworkClassifier.layers
else:
    raise RuntimeError("Unsupported model type.")

# Add a new layer: food_prob = 1 - var_1626
from coremltools.proto import NeuralNetwork_pb2
invert_layer = NeuralNetwork_pb2.NeuralNetworkLayer()
invert_layer.name = "invert_output"
invert_layer.input.append(output_name)
invert_layer.output.append("food_prob")
invert_layer.activation.linear.alpha = -1.0
invert_layer.activation.linear.beta = 1.0
layers.append(invert_layer)

# Update output description
spec.description.output[0].name = "food_prob"
spec.description.output[0].shortDescription = "Probability of food (correct polarity)"

print(f"Saving patched model to: {out_path}")
ct.models.utils.save_spec(spec, str(out_path))
print("Done. Use FoodDetector_fixed.mlmodel for correct output polarity.")
