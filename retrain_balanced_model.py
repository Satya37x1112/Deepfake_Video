#!/usr/bin/env python3
"""
RETRAIN MODEL WITH BALANCED DATASET
Fixes the issue where model detects all videos as FAKE
"""

import os
import json
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Define constants (same as training)
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
    """Build sequence model for deepfake detection"""
    
    # Feature input
    frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
    
    # Use GRU layers for temporal analysis
    x = tf.keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = tf.keras.layers.GRU(8)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model([frame_features_input, mask_input], output)
    
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    return model

def load_balanced_dataset():
    """Load dataset with BALANCED classes (equal REAL and FAKE videos)"""
    
    # Paths to organized videos
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
    
    # Balance the dataset by limiting FAKE videos to match REAL videos
    num_real = len(real_videos)
    num_fake_to_use = num_real  # Use same number of fake videos as real
    
    if len(fake_videos) > num_fake_to_use:
        print(f"\n[BALANCING] Using {num_fake_to_use} FAKE videos to balance with {num_real} REAL videos")
        # Randomly sample fake videos
        np.random.seed(42)
        fake_videos = list(np.random.choice(fake_videos, num_fake_to_use, replace=False))
    
    # Add REAL videos
    for video_path in real_videos:
        video_paths.append(str(video_path))
        labels.append(0)  # REAL = 0
    
    # Add FAKE videos
    for video_path in fake_videos:
        video_paths.append(str(video_path))
        labels.append(1)  # FAKE = 1
    
    print(f"\n[BALANCED DATASET]")
    print(f"Total videos: {len(video_paths)}")
    print(f"REAL videos: {sum(1 for l in labels if l == 0)} ({sum(1 for l in labels if l == 0)/len(labels)*100:.1f}%)")
    print(f"FAKE videos: {sum(1 for l in labels if l == 1)} ({sum(1 for l in labels if l == 1)/len(labels)*100:.1f}%)")
    
    return video_paths, labels

def plot_training_history(history, save_path='training_history_balanced.png'):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy (Balanced Dataset)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss (Balanced Dataset)')
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
    print("BALANCED DeepFake Detection Model Training")
    print("Fixing: Model detects all videos as FAKE")
    print("Solution: Train with equal REAL and FAKE videos")
    print("=" * 60)
    
    # Load BALANCED dataset
    video_paths, labels = load_balanced_dataset()
    
    if len(video_paths) == 0:
        print("Error: No videos to train on!")
        return
    
    # Prepare features
    print("\nExtracting features from videos...")
    features, masks, labels = prepare_all_videos(video_paths, labels)
    
    if len(features) == 0:
        print("Error: No features extracted!")
        return
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Compute class weights (for extra safety)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=labels
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\nClass weights: REAL={class_weights[0]:.2f}, FAKE={class_weights[1]:.2f}")
    
    # Split data
    train_features, val_features, train_masks, val_masks, train_labels, val_labels = train_test_split(
        features, masks, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(train_features)}")
    print(f"  - REAL: {sum(train_labels == 0)}")
    print(f"  - FAKE: {sum(train_labels == 1)}")
    print(f"Validation samples: {len(val_features)}")
    print(f"  - REAL: {sum(val_labels == 0)}")
    print(f"  - FAKE: {sum(val_labels == 1)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./model/deepfake_video_model_balanced_best.h5",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model WITH CLASS WEIGHTS
    print("\nTraining model with balanced dataset and class weights...")
    history = model.fit(
        [train_features, train_masks],
        train_labels,
        validation_data=([val_features, val_masks], val_labels),
        epochs=50,
        batch_size=8,
        class_weight=class_weight_dict,  # IMPORTANT: Use class weights
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
    
    # Calculate per-class accuracy
    real_accuracy = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
    fake_accuracy = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
    
    print(f"\nPer-class Performance:")
    print(f"REAL video detection: {real_accuracy:.2f}%")
    print(f"FAKE video detection: {fake_accuracy:.2f}%")
    
    # Backup old model
    old_model_path = "./model/deepfake_video_model.h5"
    if os.path.exists(old_model_path):
        backup_path = "./model/deepfake_video_model_OLD_UNBALANCED.h5"
        os.rename(old_model_path, backup_path)
        print(f"\n[BACKUP] Old model backed up to: {backup_path}")
    
    # Save new balanced model
    model.save("./model/deepfake_video_model.h5")
    print("\n[SUCCESS] New BALANCED model saved to ./model/deepfake_video_model.h5")
    
    # Plot history
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("BALANCED MODEL TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Real video accuracy: {real_accuracy:.2f}%")
    print(f"Fake video accuracy: {fake_accuracy:.2f}%")
    print("\n[NEXT STEP] Restart your Flask app to use the new balanced model")
    print("=" * 60)

if __name__ == "__main__":
    main()
