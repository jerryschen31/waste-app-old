#!/usr/bin/env python3
"""Fine-tune EfficientNet-B2 as a binary (food / not_food) classifier.

Usage (example):
  source ml_env/bin/activate
  python3 scripts/train_food_binary.py --data_dir data --checkpoint_dir outputs/efficientnet_b2_ft --input_size 224 --batch_size 32 --epochs 12

This script:
- loads images from directory structure: data/train/food, data/train/not_food, etc.
- builds an EfficientNetB2 backbone and a binary head
- attempts to load `models/efficientnet-b2/model.weights.h5` if present
- trains and saves best checkpoint and SavedModel
"""

import argparse
import os
import sys
from pathlib import Path

import tensorflow as tf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Root data directory with train/val/test subfolders")
    p.add_argument("--checkpoint_dir", required=True, help="Directory to save checkpoints and outputs")
    p.add_argument("--input_size", type=int, default=224, help="Input image size (square)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--initial_lr", type=float, default=1e-4)
    p.add_argument("--weights_path", default="models/efficientnet-b2/model.weights.h5", help="Optional local weights file to initialize backbone")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_datasets(data_dir, input_size, batch_size, seed=42):
    AUTOTUNE = tf.data.AUTOTUNE
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        image_size=(input_size, input_size),
        batch_size=batch_size,
        seed=seed,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        image_size=(input_size, input_size),
        batch_size=batch_size,
        seed=seed + 1,
    )

    test_ds = None
    if os.path.isdir(test_dir):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            image_size=(input_size, input_size),
            batch_size=batch_size,
            seed=seed + 2,
        )

    # Prefetch and simple optimization
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model(input_size, weights_path=None, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))

    # Data augmentation (on GPU if available)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.08),
    ], name="augmentation")

    x = aug(inputs)

    # Try to use Keras built-in EfficientNetB2; if custom weights present, load them.
    try:
        base = tf.keras.applications.EfficientNetB2(include_top=False, weights=None, input_tensor=x, pooling=None)
    except Exception:
        # fallback: construct model without input tensor
        base = tf.keras.applications.EfficientNetB2(include_top=False, weights=None, input_shape=(input_size, input_size, 3))
        x = aug(base.input)
        base = tf.keras.applications.EfficientNetB2(include_top=False, weights=None, input_tensor=x, pooling=None)

    # Load local weights if available (best-effort)
    if weights_path and os.path.exists(weights_path):
        try:
            # try full-weights load
            base.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print(f"Loaded local weights from {weights_path} (by_name, skip_mismatch=True)")
        except Exception as e:
            print("Warning: could not load local weights cleanly:", e)

    # Build head
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    if dropout_rate and dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="effnet_b2_binary")
    return model


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds, val_ds, test_ds = make_datasets(args.data_dir, args.input_size, args.batch_size, seed=args.seed)

    model = build_model(args.input_size, weights_path=args.weights_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.initial_lr)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Save final artifacts
    final_h5 = os.path.join(args.checkpoint_dir, "final_model.h5")
    model.save(final_h5)
    print("Saved final model to", final_h5)

    saved_model_dir = os.path.join(args.checkpoint_dir, "saved_model")
    model.save(saved_model_dir, include_optimizer=False)
    print("Saved SavedModel to", saved_model_dir)

    # Evaluate on test set if present
    if test_ds is not None:
        results = model.evaluate(test_ds)
        print("Test results:", results)


if __name__ == "__main__":
    main()
