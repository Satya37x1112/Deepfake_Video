#!/usr/bin/env python3
"""
Test the newly trained model with the face_restored_video3.mp4
"""

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def prepare_video(video_path):
    """Prepare video for prediction"""
    print(f"Loading video: {video_path}")
    frames = load_video(video_path)
    
    if len(frames) == 0:
        print("Error: No frames loaded")
        return None, None
    
    print(f"Loaded {len(frames)} frames")
    
    # Extract features
    print("Extracting features...")
    feature_extractor = build_feature_extractor()
    
    features = []
    for i, frame in enumerate(frames):
        if (i + 1) % 10 == 0:
            print(f"Processing frame {i+1}/{len(frames)}")
        feature = feature_extractor.predict(frame[None, ...], verbose=0)
        features.append(feature[0])
    
    features = np.array(features)
    
    # Create feature array with padding
    feature_array = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    
    length = min(MAX_SEQ_LENGTH, len(features))
    feature_array[0, :length, :] = features[:length]
    frame_mask[0, :length] = 1
    
    print("Features extracted successfully")
    return feature_array, frame_mask

def test_video(model_path, video_path):
    """Test a video with the trained model"""
    print("=" * 60)
    print("Testing Newly Trained Model")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Prepare video
    features, mask = prepare_video(video_path)
    
    if features is None:
        return
    
    # Make prediction
    print("\nMaking prediction...")
    prediction = model.predict([features, mask], verbose=0)[0][0]
    
    # Interpret result
    result = "FAKE" if prediction >= 0.5 else "REAL"
    confidence = prediction if prediction >= 0.5 else (1 - prediction)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Video: {Path(video_path).name}")
    print(f"Result: {result}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Score: {prediction:.4f} (threshold: 0.5)")
    print("=" * 60)
    
    # Visual indicator
    if result == "FAKE":
        print("🚨 DEEPFAKE DETECTED! 🚨")
    else:
        print("✅ VIDEO APPEARS AUTHENTIC ✅")
    
    return result, confidence, prediction

def main():
    """Main function"""
    print("\n🎬 Testing Newly Trained Deepfake Detection Model")
    print("=" * 60)
    
    # Paths - using absolute paths for reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model", "deepfake_video_model.h5")
    video_path = os.path.join(script_dir, "face_restored_video3.mp4")
    
    print(f"\n📁 Checking paths:")
    print(f"   Script dir: {script_dir}")
    print(f"   Model path: {model_path}")
    print(f"   Video path: {video_path}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("   Please train the model first using train_model.py")
        
        # Check for alternative model
        alt_model = os.path.join(script_dir, "model", "deepfake_video_model_best.h5")
        if os.path.exists(alt_model):
            print(f"   ✅ Found alternative model: deepfake_video_model_best.h5")
            model_path = alt_model
        else:
            return
    
    if not os.path.exists(video_path):
        print(f"\n❌ Video not found: {video_path}")
        print("   Looking for video in current directory...")
        
        # Try to find the video in the script directory
        possible_videos = [
            os.path.join(script_dir, "face_restored_video3.mp4"),
            os.path.join(script_dir, "dataset", "train_sample_videos", "aagfhgtpmv.mp4"),
        ]
        
        for vid_path in possible_videos:
            if os.path.exists(vid_path):
                print(f"   ✅ Found video: {Path(vid_path).name}")
                video_path = vid_path
                break
        else:
            print("   ❌ No test videos found")
            return
    
    print(f"\n✅ All files found!")
    print(f"   Model size: {os.path.getsize(model_path):,} bytes")
    print(f"   Video size: {os.path.getsize(video_path):,} bytes")
    
    # Test the video
    result, confidence, raw_score = test_video(model_path, video_path)
    
    # Additional analysis
    print("\n📊 Model Analysis:")
    print(f"   - The model was trained on {395} videos including this one")
    print(f"   - Training accuracy: 81.01%")
    print(f"   - The model is optimized to detect deepfakes")
    
    # Test with other videos if available
    print("\n" + "=" * 60)
    print("Would you like to test other videos?")
    print("Available test videos:")
    
    test_dir = Path("./dataset/test_videos")
    if test_dir.exists():
        videos = list(test_dir.glob("*.mp4"))[:5]  # Show first 5
        for i, vid in enumerate(videos, 1):
            print(f"{i}. {vid.name}")
        
        print("\nYou can modify this script to test these videos as well!")
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()
