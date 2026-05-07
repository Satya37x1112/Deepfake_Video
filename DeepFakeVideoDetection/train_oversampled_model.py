#!/usr/bin/env python3
"""
RETRAIN WITH AGGRESSIVE OVERSAMPLING
Strategy: Oversample REAL videos to match FAKE count (320 each)
This will force the model to learn REAL patterns better
"""

import os
import json
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Define constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 30
NUM_FEATURES = 2048

def crop_center_square(frame):
    """Crop the center square of a video frame"""
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    """Load and preprocess video"""
    cap = cv2.VideoCapture(path)
    frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    
    return np.array(frames)

def build_feature_extractor():
    """Build InceptionV3 feature extractor"""
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

def prepare_all_videos(video_paths, labels):
    """Prepare all videos by extracting features"""
    print(f"Extracting features from {len(video_paths)} videos...")
    
    feature_extractor = build_feature_extractor()
    
    all_features = []
    all_labels = []
    frame_masks = []
    
    for idx, (video_path, label) in enumerate(zip(video_paths, labels)):
        try:
            # Load video
            frames = load_video(video_path)
            
            if len(frames) == 0:
                print(f"Skipping {video_path} - no frames loaded")
                continue
            
            # Extract features
            features = []
            for frame in frames:
                feature = feature_extractor.predict(frame[None, ...], verbose=0)
                features.append(feature[0])
            
            features = np.array(features)
            
            # Create feature array
            feature_array = np.zeros(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
            frame_mask = np.zeros(shape=(MAX_SEQ_LENGTH,), dtype="bool")
            
            length = min(MAX_SEQ_LENGTH, len(features))
            feature_array[:length, :] = features[:length]
            frame_mask[:length] = 1  # 1 = not masked
            
            all_features.append(feature_array)
            frame_masks.append(frame_mask)
            all_labels.append(label)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(video_paths)} videos")
                
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    return np.array(all_features), np.array(frame_masks), np.array(all_labels)

def build_model():
    """Build model with INCREASED capacity"""
    
    # Feature input
    frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
    
    # Deeper GRU layers
    x = tf.keras.layers.GRU(128, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.GRU(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.GRU(32)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model([frame_features_input, mask_input], output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    return model

def load_oversampled_dataset():
    """Load dataset with AGGRESSIVE OVERSAMPLING of REAL videos"""
    
    real_folder = Path("dataset/train_sample_videos/real")
    fake_folder = Path("dataset/train_sample_videos/fake")
    
    video_paths = []
    labels = []
    
    # Load REAL videos
    real_videos = list(real_folder.glob("*.mp4"))
    print(f"Found {len(real_videos)} REAL videos")
    
    # Load FAKE videos
    fake_videos = list(fake_folder.glob("*.mp4"))
    print(f"Found {len(fake_videos)} FAKE videos")
    
    # OVERSAMPLE REAL videos to match FAKE count
    num_fake = len(fake_videos)
    num_real = len(real_videos)
    
    print(f"\n[OVERSAMPLING STRATEGY]")
    print(f"Original: {num_real} REAL, {num_fake} FAKE")
    print(f"Target: {num_fake} REAL, {num_fake} FAKE (50-50 balance)")
    
    # Calculate how many times to repeat each real video
    repeats_needed = num_fake // num_real
    extra_samples = num_fake % num_real
    
    print(f"Will repeat each REAL video {repeats_needed} times")
    print(f"Plus {extra_samples} random REAL videos for exact balance")
    
    # Add REAL videos (with repetition)
    for _ in range(repeats_needed):
        for video_path in real_videos:
            video_paths.append(str(video_path))
            labels.append(0)  # REAL = 0
    
    # Add extra samples
    np.random.seed(42)
    extra_real_videos = np.random.choice(real_videos, extra_samples, replace=False)
    for video_path in extra_real_videos:
        video_paths.append(str(video_path))
        labels.append(0)
    
    # Add ALL FAKE videos
    for video_path in fake_videos:
        video_paths.append(str(video_path))
        labels.append(1)  # FAKE = 1
    
    print(f"\n[BALANCED DATASET WITH OVERSAMPLING]")
    print(f"Total videos: {len(video_paths)}")
    print(f"REAL videos (oversampled): {sum(1 for l in labels if l == 0)} ({sum(1 for l in labels if l == 0)/len(labels)*100:.1f}%)")
    print(f"FAKE videos: {sum(1 for l in labels if l == 1)} ({sum(1 for l in labels if l == 1)/len(labels)*100:.1f}%)")
    
    # Shuffle
    indices = np.random.permutation(len(video_paths))
    video_paths = [video_paths[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return video_paths, labels

def plot_training_history(history, save_path='training_history_oversampled.png'):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy (Oversampled Dataset)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss (Oversampled Dataset)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")
    plt.close()

def main():
    """Main training function"""
    print("=" * 60)
    print("OVERSAMPLED DeepFake Detection Model Training")
    print("Strategy: Oversample REAL videos to 320 (match FAKE count)")
    print("Goal: HIGH CONFIDENCE on both REAL and FAKE")
    print("=" * 60)
    
    # Load OVERSAMPLED dataset
    video_paths, labels = load_oversampled_dataset()
    
    if len(video_paths) == 0:
        print("Error: No videos to train on!")
        return
    
    # Prepare features
    print("\nExtracting features from videos (this will take time)...")
    features, masks, labels = prepare_all_videos(video_paths, labels)
    
    if len(features) == 0:
        print("Error: No features extracted!")
        return
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split data
    train_features, val_features, train_masks, val_masks, train_labels, val_labels = train_test_split(
        features, masks, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(train_features)}")
    print(f"  - REAL: {sum(train_labels == 0)}")
    print(f"  - FAKE: {sum(train_labels == 1)}")
    print(f"Validation samples: {len(val_features)}")
    print(f"  - REAL: {sum(val_labels == 0)}")
    print(f"  - FAKE: {sum(val_labels == 1)}")
    
    # Build model
    print("\nBuilding deeper model...")
    model = build_model()
    model.summary()
    
    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./model/deepfake_video_model_oversampled_best.h5",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\nTraining with BALANCED (oversampled) dataset...")
    history = model.fit(
        [train_features, train_masks],
        train_labels,
        validation_data=([val_features, val_masks], val_labels),
        epochs=100,
        batch_size=16,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    val_predictions = model.predict([val_features, val_masks])
    val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_predictions_binary, 
                                target_names=['REAL', 'FAKE']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels, val_predictions_binary)
    print("           Predicted")
    print("         REAL  FAKE")
    print(f"REAL    {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"FAKE    {cm[1,0]:5d} {cm[1,1]:5d}")
    
    # Calculate metrics
    real_accuracy = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
    fake_accuracy = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
    
    real_predictions = val_predictions[val_labels == 0]
    fake_predictions = val_predictions[val_labels == 1]
    
    avg_real_confidence = (1 - np.mean(real_predictions)) * 100
    avg_fake_confidence = np.mean(fake_predictions) * 100
    
    print(f"\nPer-class Performance:")
    print(f"REAL video detection: {real_accuracy:.2f}% (avg confidence: {avg_real_confidence:.2f}%)")
    print(f"FAKE video detection: {fake_accuracy:.2f}% (avg confidence: {avg_fake_confidence:.2f}%)")
    
    # Backup and save
    old_model_path = "./model/deepfake_video_model.h5"
    if os.path.exists(old_model_path):
        backup_path = "./model/deepfake_video_model_BEFORE_OVERSAMPLE.h5"
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(old_model_path, backup_path)
        print(f"\n[BACKUP] Previous model → {backup_path}")
    
    model.save("./model/deepfake_video_model.h5")
    print("\n[SUCCESS] New OVERSAMPLED model saved!")
    
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("OVERSAMPLED MODEL TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Real detection: {real_accuracy:.2f}% (confidence: {avg_real_confidence:.2f}%)")
    print(f"Fake detection: {fake_accuracy:.2f}% (confidence: {avg_fake_confidence:.2f}%)")
    print("\n[NEXT] Test with: python test_real_videos.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
