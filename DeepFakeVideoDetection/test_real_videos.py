"""
Test Real Video Detection - Diagnose why real videos are being misclassified
"""
import numpy as np
import tensorflow as tf
import cv2
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Configuration
IMG_SIZE = 224
MAX_SEQ_LENGTH = 30  # MUST match training value (model expects 30 frames)
NUM_FEATURES = 2048

def build_feature_extractor():
    """Build InceptionV3 feature extractor"""
    inception_base = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = inception_base(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

def load_video(video_path, max_frames=30):  # MUST match MAX_SEQ_LENGTH
    """Load and process video frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Pad if necessary
    while len(frames) < max_frames:
        frames.append(frames[-1])
    
    return np.array(frames[:max_frames])

def test_real_videos():
    """Test real videos from the dataset"""
    
    base_dir = Path(__file__).parent
    model_path = base_dir / "model" / "deepfake_video_model.h5"
    real_videos_dir = base_dir / "dataset" / "train_sample_videos" / "real"
    
    print("="*60)
    print("REAL VIDEO DETECTION TEST")
    print("="*60)
    
    # Load model
    print("\n[LOADING] Model...")
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        return
    
    model = tf.keras.models.load_model(str(model_path))
    print(f"[SUCCESS] Model loaded from {model_path}")
    
    # Load feature extractor
    print("[LOADING] Feature extractor...")
    feature_extractor = build_feature_extractor()
    print("[SUCCESS] Feature extractor loaded")
    
    # Get real videos
    if not real_videos_dir.exists():
        print(f"[ERROR] Real videos directory not found: {real_videos_dir}")
        return
    
    real_videos = list(real_videos_dir.glob("*.mp4"))
    
    if len(real_videos) == 0:
        print("[ERROR] No real videos found!")
        return
    
    print(f"\n[INFO] Found {len(real_videos)} real videos")
    print(f"[INFO] Testing first 10 videos...\n")
    
    # Test statistics
    correct_detections = 0
    incorrect_detections = 0
    results = []
    
    # Test first 10 videos
    for idx, video_path in enumerate(real_videos[:10], 1):
        try:
            print(f"[{idx}/10] Testing: {video_path.name}")
            
            # Load video
            frames = load_video(str(video_path), MAX_SEQ_LENGTH)
            
            if frames is None:
                print(f"  [ERROR] Failed to load video")
                continue
            
            # Extract features
            features = feature_extractor.predict(frames, verbose=0)
            
            # Create mask (matching training format)
            frame_mask = np.zeros(shape=(MAX_SEQ_LENGTH,), dtype="bool")
            feature_array = np.zeros(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
            
            length = min(MAX_SEQ_LENGTH, len(features))
            feature_array[:length, :] = features[:length]
            frame_mask[:length] = 1  # 1 = not masked
            
            # Expand dimensions for batch prediction
            feature_array = np.expand_dims(feature_array, axis=0)
            frame_mask = np.expand_dims(frame_mask, axis=0)
            
            # Predict with both inputs
            prediction = model.predict([feature_array, frame_mask], verbose=0)[0][0]
            prediction_percent = prediction * 100
            
            # Interpret result
            if prediction < 0.5:
                result = "REAL"
                correct = True
                correct_detections += 1
            else:
                result = "FAKE"
                correct = False
                incorrect_detections += 1
            
            status = "[CORRECT]" if correct else "[WRONG]"
            print(f"  {status} Prediction: {result} ({prediction_percent:.2f}%)")
            
            results.append({
                'video': video_path.name,
                'prediction': prediction,
                'result': result,
                'correct': correct
            })
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tested: {len(results)}")
    print(f"Correct (Detected as REAL): {correct_detections}")
    print(f"Incorrect (Detected as FAKE): {incorrect_detections}")
    
    if len(results) > 0:
        accuracy = (correct_detections / len(results)) * 100
        print(f"Accuracy on REAL videos: {accuracy:.2f}%")
    
    # Show details of misclassified videos
    if incorrect_detections > 0:
        print("\n[WARNING] Misclassified REAL videos (detected as FAKE):")
        for r in results:
            if not r['correct']:
                print(f"  - {r['video']}: {r['prediction']*100:.2f}% (threshold: 50%)")
    
    # Show prediction distribution
    print("\n[INFO] Prediction Distribution:")
    predictions = [r['prediction'] for r in results]
    if predictions:
        print(f"  Min: {min(predictions)*100:.2f}%")
        print(f"  Max: {max(predictions)*100:.2f}%")
        print(f"  Avg: {np.mean(predictions)*100:.2f}%")
        print(f"  Median: {np.median(predictions)*100:.2f}%")
    
    print("\n[INFO] Prediction threshold: 50% (above = FAKE, below = REAL)")
    print("="*60)
    
    return results

if __name__ == "__main__":
    try:
        results = test_real_videos()
    except KeyboardInterrupt:
        print("\n[CANCELLED] Test cancelled by user")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
