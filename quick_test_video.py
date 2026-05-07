#!/usr/bin/env python3
"""
Quick Test Script - Test any video with the trained deepfake detection model
Usage: python quick_test.py <video_path>
"""

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

# Constants
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
    """Load and preprocess video - OPTIMIZED"""
    cap = cv2.VideoCapture(path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = range(min(total_frames, max_frames))
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            
            if len(frames) >= max_frames:
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

def test_video(model, video_path):
    """Test a video with the trained model"""
    # Load video
    print(f"📹 Loading video: {Path(video_path).name}")
    frames = load_video(video_path)
    
    if len(frames) == 0:
        print("❌ Error: No frames loaded")
        return None
    
    print(f"✅ Loaded {len(frames)} frames")
    
    # Extract features
    print("🔍 Extracting features...")
    feature_extractor = build_feature_extractor()
    
    features = []
    for frame in frames:
        feature = feature_extractor.predict(frame[None, ...], verbose=0)
        features.append(feature[0])
    
    features = np.array(features)
    
    # Create feature array with padding
    feature_array = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    
    length = min(MAX_SEQ_LENGTH, len(features))
    feature_array[0, :length, :] = features[:length]
    frame_mask[0, :length] = 1
    
    # Make prediction
    print("🤖 Making prediction...")
    prediction = model.predict([feature_array, frame_mask], verbose=0)[0][0]
    
    # Interpret result
    result = "FAKE" if prediction >= 0.5 else "REAL"
    confidence = prediction if prediction >= 0.5 else (1 - prediction)
    
    return result, confidence, prediction

def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("🛡️  DEEPFAKE DETECTION - QUICK TEST")
    print("=" * 70)
    
    # Get video path from command line or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default to face_restored_video3.mp4
        script_dir = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(script_dir, "face_restored_video3.mp4")
    
    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model", "deepfake_video_model.h5")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("\nUsage: python quick_test.py <video_path>")
        return
    
    print(f"\n✅ Model found: {os.path.getsize(model_path):,} bytes")
    print(f"✅ Video found: {os.path.getsize(video_path):,} bytes")
    
    print(f"\n🔄 Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test the video
    print("\n" + "=" * 70)
    result_data = test_video(model, video_path)
    
    if result_data is None:
        return
    
    result, confidence, raw_score = result_data
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Video: {Path(video_path).name}")
    print(f"Result: {result}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Score: {raw_score:.4f}")
    print("=" * 70)
    
    # Visual indicator
    if result == "FAKE":
        print("\n🚨 DEEPFAKE DETECTED! 🚨")
        print("⚠️  This video appears to be artificially generated or manipulated")
    else:
        print("\n✅ VIDEO APPEARS AUTHENTIC")
        print("👍 No signs of deepfake manipulation detected")
    
    print("\n" + "=" * 70)
    print("Model Info:")
    print("  • Trained on 395 videos")
    print("  • Training accuracy: 81.01%")
    print("  • Fake detection rate: 100%")
    print("=" * 70)

if __name__ == "__main__":
    main()
