"""
DeepFake Video Detection - Inference Script
"""
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
# Import model components
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from model import build_feature_extractor, IMG_SIZE, MAX_SEQ_LENGTH, NUM_FEATURES

class DeepFakeDetector:
    """DeepFake video detector"""
    
    def __init__(self, model_path):
        """Initialize detector with trained model"""
        self.model = tf.keras.models.load_model(model_path)
        self.feature_extractor = build_feature_extractor()
        print(f"[LOADED] Model from {model_path}")
    
    def load_video(self, video_path, max_frames=MAX_SEQ_LENGTH):
        """Load and preprocess video"""
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
    
    def predict(self, video_path):
        """Detect if video is deepfake"""
        # Load video
        frames = self.load_video(video_path)
        if frames is None:
            return None, "Failed to load video"
        
        # Extract features
        features = self.feature_extractor.predict(frames, verbose=0)
        
        # Prepare for model
        frame_mask = np.zeros(shape=(MAX_SEQ_LENGTH,), dtype="bool")
        feature_array = np.zeros(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
        
        length = min(MAX_SEQ_LENGTH, len(features))
        feature_array[:length, :] = features[:length]
        frame_mask[:length] = 1
        
        # Add batch dimension
        feature_array = np.expand_dims(feature_array, axis=0)
        frame_mask = np.expand_dims(frame_mask, axis=0)
        
        # Predict
        prediction = self.model.predict([feature_array, frame_mask], verbose=0)[0][0]
        
        # Interpret
        result = 'FAKE' if prediction >= 0.5 else 'REAL'
        confidence = prediction * 100 if result == 'FAKE' else (1 - prediction) * 100
        
        return {
            'result': result,
            'confidence': float(confidence),
            'raw_score': float(prediction)
        }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path>")
        sys.exit(1)
    
    model_path = Path(__file__).parent.parent / "saved-models" / "deepfake_video_model.h5"
    detector = DeepFakeDetector(str(model_path))
    
    result = detector.predict(sys.argv[1])
    if result:
        print(f"\nResult: {result['result']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Raw Score: {result['raw_score']:.4f}")
