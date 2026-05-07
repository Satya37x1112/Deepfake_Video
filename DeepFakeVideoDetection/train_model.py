#!/usr/bin/env python3
"""
Training script for deepfake detection model
Trains the model with existing dataset and new videos
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

# Define constants (same as app.py)
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

def load_dataset_with_new_video(new_video_path=None, new_video_label="FAKE"):
    """Load dataset including new video if provided"""
    
    # Load metadata
    metadata_path = Path("dataset/train_sample_videos/metadata.json")
    video_folder = Path("dataset/train_sample_videos")
    
    video_paths = []
    labels = []
    
    # Add new video if provided
    if new_video_path and os.path.exists(new_video_path):
        print(f"Adding new video: {new_video_path}")
        video_paths.append(new_video_path)
        labels.append(1 if new_video_label == "FAKE" else 0)
        print(f"Label: {new_video_label} ({labels[-1]})")
    
    # Load existing dataset
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        for filename, info in metadata.items():
            video_path = video_folder / filename
            if video_path.exists():
                video_paths.append(str(video_path))
                # FAKE = 1, REAL = 0
                labels.append(1 if info['label'] == 'FAKE' else 0)
    
    print(f"Total videos to process: {len(video_paths)}")
    print(f"FAKE videos: {sum(labels)}")
    print(f"REAL videos: {len(labels) - sum(labels)}")
    
    return video_paths, labels

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
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
    print("DeepFake Detection Model Training")
    print("=" * 60)
    
    # Path to the new deepfake video
    new_video_path = r"C:\Users\manoh\OneDrive\Desktop\DeepFakeVideoDetection\DeepFakeVideoDetection\face_restored_video3.mp4"
    
    # Check if new video exists
    if not os.path.exists(new_video_path):
        print(f"Warning: New video not found at {new_video_path}")
        print("Training with existing dataset only...")
        new_video_path = None
    else:
        print(f"New deepfake video found: {new_video_path}")
    
    # Load dataset
    video_paths, labels = load_dataset_with_new_video(
        new_video_path=new_video_path,
        new_video_label="FAKE"
    )
    
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
    
    # Split data
    train_features, val_features, train_masks, val_masks, train_labels, val_labels = train_test_split(
        features, masks, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./model/deepfake_video_model_best.h5",
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
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        [train_features, train_masks],
        train_labels,
        validation_data=([val_features, val_masks], val_labels),
        epochs=50,
        batch_size=8,
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
    print(cm)
    
    # Save final model
    model.save("./model/deepfake_video_model.h5")
    print("\nModel saved to ./model/deepfake_video_model.h5")
    
    # Plot history
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
