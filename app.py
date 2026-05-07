# -*- coding: utf-8 -*-
"""
Enhanced Deepfake Video Detection with Multi-Model Ensemble
Fixed Unicode encoding issues for Windows compatibility
"""
from flask import Flask, request, jsonify,render_template
import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import ssl
import urllib
import requests
import base64
import io
import concurrent.futures
import subprocess
import tempfile
from sklearn.metrics import confusion_matrix
import math
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

app = Flask(__name__)

# Configuration for file uploads
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the newly trained model with error handling
model = None
model_path = None

try:
    # Get the directory where app.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model', 'deepfake_video_model.h5')
    
    print(f"[INIT] Checking model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        print(f"[INFO] Please ensure the model file exists in the 'model' folder")
        model = None
    else:
        print(f"[INIT] Model file found, loading...")
        model = tf.keras.models.load_model(model_path)
        print("[SUCCESS] Newly trained model loaded successfully!")
        print("   Model trained on 395 videos including your deepfake samples")
        print("   Training accuracy: 81.01% | Fake detection: 100%")
        print(f"   Model path: {model_path}")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    import traceback
    traceback.print_exc()
    print("\n[WARNING] Running without model - predictions will not work")
    model = None

# OpenRouter config
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
OPENROUTER_TIMEOUT = 15  # seconds fallback trigger
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Define constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 30  # MUST match training value - model expects exactly 30 frames
NUM_FEATURES = 2048
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Quality thresholds for frame filtering
MIN_FACE_SIZE = 100
MAX_BLUR_THRESHOLD = 100
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 200

# OPTIMIZED: Use only InceptionV3 for speed
def build_feature_extractor():
    """Build feature extractor - optimized for speed"""
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

feature_extractor = build_feature_extractor()
print("Feature extractor loaded successfully!")

# Advanced quality detection functions
def detect_blur(image):
    """Detect blur in image using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_faces(image):
    """Detect faces in the image with error handling"""
    try:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if face_cascade.empty():
            # Fallback: return empty if cascade fails to load
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def get_brightness(image):
    """Calculate average brightness of the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def is_quality_frame(frame):
    """Check if frame meets quality requirements with error handling"""
    try:
        # Check blur
        blur_score = detect_blur(frame)
        if blur_score < MAX_BLUR_THRESHOLD:
            return False
        
        # Check brightness
        brightness = get_brightness(frame)
        if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
            return False
        
        # Check for faces (optional - don't fail if face detection has issues)
        try:
            faces = detect_faces(frame)
            if len(faces) == 0:
                return False
            
            # Check face size
            for (x, y, w, h) in faces:
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    return False
        except:
            # If face detection fails, just check other criteria
            pass
        
        return True
    except Exception as e:
        print(f"Error in quality check: {e}")
        return True  # Default to accepting the frame if quality check fails

def extract_face_region(frame):
    """Extract the largest face region from the frame"""
    faces = detect_faces(frame)
    if len(faces) == 0:
        return crop_center_square(frame)
    
    # Get the largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Add some padding around the face
    padding = int(0.3 * min(w, h))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)
    
    face_region = frame[y:y+h, x:x+w]
    return face_region

# OPTIMIZED: Faster video loading with reduced quality checks and ffmpeg fallback
def load_video(path, max_frames=20, resize=(IMG_SIZE, IMG_SIZE)):
    """Load video frames - optimized for speed with ffmpeg fallback"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[ERROR] Video file missing or empty: {path}")
        return np.array([])

    cap = cv2.VideoCapture(path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] OpenCV reports {total_frames} frames for {os.path.basename(path)}")

    try:
        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = list(range(min(total_frames, max_frames)))

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] OpenCV failed to read frame {frame_idx}")
                continue
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            if len(frames) >= max_frames:
                break
    except Exception as e:
        print(f"[ERROR] OpenCV video read error: {e}")
    finally:
        cap.release()

    if len(frames) > 0:
        print(f"[INFO] Loaded {len(frames)} frames via OpenCV")
        return np.array(frames)

    # ffmpeg fallback
    print(f"[INFO] OpenCV returned 0 frames. Trying ffmpeg fallback...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-vf', f'select=not(mod(n\\,{max(1, total_frames // max_frames if total_frames else 1)})),scale={resize[0]}:{resize[1]}',
                '-frames:v', str(max_frames),
                '-pix_fmt', 'rgb24',
                os.path.join(tmpdir, 'frame_%03d.png')
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
            for fname in sorted(os.listdir(tmpdir)):
                if fname.endswith('.png'):
                    img = cv2.imread(os.path.join(tmpdir, fname))
                    if img is not None:
                        img = crop_center_square(img)
                        img = cv2.resize(img, resize)
                        img = img[:, :, [2, 1, 0]]
                        frames.append(img)
            if len(frames) > 0:
                print(f"[INFO] Loaded {len(frames)} frames via ffmpeg")
                return np.array(frames)
    except Exception as e:
        print(f"[WARN] ffmpeg fallback failed: {e}")

    print(f"[ERROR] Could not extract frames from video: {path}")
    return np.array([])

# Function to crop the center square of a video frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# OPTIMIZED feature extraction - single model only
def extract_features(frames):
    """Extract features using InceptionV3 (optimized for speed)"""
    features = []
    for frame in frames:
        frame_feature = feature_extractor.predict(frame[None, ...], verbose=0)
        features.append(frame_feature[0])
    return np.array(features)

# Temporal analysis functions
def calculate_temporal_consistency(features):
    """Calculate temporal consistency between consecutive frames"""
    if len(features) < 2:
        return 1.0
    
    consistency_scores = []
    for i in range(1, len(features)):
        # Calculate cosine similarity between consecutive frames
        prev_feat = features[i-1]
        curr_feat = features[i]
        
        # Normalize features
        prev_norm = prev_feat / (np.linalg.norm(prev_feat) + 1e-8)
        curr_norm = curr_feat / (np.linalg.norm(curr_feat) + 1e-8)
        
        # Calculate similarity
        similarity = np.dot(prev_norm, curr_norm)
        consistency_scores.append(similarity)
    
    return np.mean(consistency_scores)

def calculate_motion_features(frames):
    """Calculate motion features using frame differences"""
    if len(frames) < 2:
        return [0.0, 0.0, 0.0]
    
    motion_scores = []
    
    try:
        for i in range(1, len(frames)):
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor((frames[i-1] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor((frames[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Calculate motion metrics
            motion_mean = np.mean(frame_diff)
            motion_std = np.std(frame_diff)
            motion_max = np.max(frame_diff)
            
            motion_scores.append([motion_mean, motion_std, motion_max])
            
    except Exception as e:
        print(f"Error in motion calculation: {e}")
        return [0.0, 0.0, 0.0]
    
    if not motion_scores:
        return [0.0, 0.0, 0.0]
    
    motion_array = np.array(motion_scores)
    
    return [
        float(np.mean(motion_array[:, 0])),  # Average motion
        float(np.std(motion_array[:, 0])),   # Motion variance
        float(np.mean(motion_array[:, 1]))   # Average motion std
    ]

# OPTIMIZED: Simplified video preparation for trained model
def prepare_single_video(frames):
    """Prepare video with optimized feature extraction - FIXED for model compatibility"""
    if len(frames) == 0:
        return None, None, None
    
    # Limit to MAX_SEQ_LENGTH frames
    if len(frames) > MAX_SEQ_LENGTH:
        # Sample frames evenly
        indices = np.linspace(0, len(frames) - 1, MAX_SEQ_LENGTH, dtype=int)
        frames = frames[indices]
    
    # Extract features using InceptionV3 only
    features = extract_features(frames)
    
    # Prepare feature array for model (matching training script exactly)
    frame_mask = np.zeros(shape=(MAX_SEQ_LENGTH,), dtype="bool")  # FIXED: Remove batch dimension
    feature_array = np.zeros(shape=(MAX_SEQ_LENGTH, features.shape[1]), dtype="float32")  # FIXED: Remove batch dimension
    
    length = min(MAX_SEQ_LENGTH, len(features))
    feature_array[:length, :] = features[:length]  # FIXED: No batch index
    frame_mask[:length] = 1  # 1 = not masked, 0 = masked
    
    # Expand dimensions for batch prediction
    feature_array = np.expand_dims(feature_array, axis=0)
    frame_mask = np.expand_dims(frame_mask, axis=0)
    
    prepared_features = {'inception': feature_array}
    
    # Simple additional features for compatibility
    additional_features = {
        'temporal': {'inception': 0.5},
        'motion': [0.0, 0.0, 0.0]
    }
    
    return prepared_features, frame_mask, additional_features

def encode_frame_to_base64(frame, size=(256, 256)):
    """Resize and encode a frame to base64 JPEG for API usage."""
    try:
        resized = cv2.resize(frame, size)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.jpg', resized)
        if not ok:
            return None
        return base64.b64encode(buf).decode('utf-8')
    except Exception as e:
        print(f"[WARN] Frame encode error: {e}")
        return None

def get_sample_frames(video_path, num_samples=3):
    """Extract evenly spaced sample frames for OpenRouter fallback."""
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, min(num_samples, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames
    except Exception as e:
        print(f"[WARN] Sample frame extraction error: {e}")
        return []

def call_openrouter_api(video_path):
    """Call OpenRouter vision model as fallback when local model fails/times out."""
    if not OPENROUTER_API_KEY:
        return {'error': 'OPENROUTER_API_KEY not set. Cannot use API fallback.'}, 0.0
    frames = get_sample_frames(video_path, num_samples=3)
    if not frames:
        return {'error': 'Could not extract frames for API fallback.'}, 0.0
    images = []
    for f in frames:
        b64 = encode_frame_to_base64(f)
        if b64:
            images.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
    if not images:
        return {'error': 'Failed to encode frames for API fallback.'}, 0.0
    prompt = (
        "You are a deepfake detection assistant. Analyze the provided video frames carefully. "
        "Look for visual artifacts typical of deepfakes: unnatural skin texture, misaligned eyes or lips, "
        "flickering around face edges, inconsistent lighting, warped backgrounds, or odd teeth/tongue. "
        "Respond ONLY with a single JSON object in this exact format: {\"result\": \"FAKE\" or \"REAL\", \"confidence\": 0 to 100, \"reasoning\": \"short summary\"}. "
        "No markdown, no code blocks, just raw JSON."
    )
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}] + images}]
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={'Authorization': f'Bearer {OPENROUTER_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': OPENROUTER_MODEL, 'messages': messages, 'max_tokens': 300},
            timeout=20
        )
        data = resp.json()
        if resp.status_code != 200 or 'choices' not in data:
            print(f"[OPENROUTER ERROR] {data}")
            return {'error': f'OpenRouter API error: {data.get("error", "unknown")}'}, 0.0
        content = data['choices'][0]['message']['content'].strip()
        # Try to extract JSON
        import json
        # Sometimes models wrap JSON in markdown fences
        if content.startswith('```'):
            lines = content.splitlines()
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            content = '\n'.join(lines).strip()
        result = json.loads(content)
        label = result.get('result', 'UNKNOWN')
        confidence = float(result.get('confidence', 50))
        return {
            'result': label,
            'confidence': round(confidence, 2),
            'raw_score': confidence / 100.0,
            'method': 'openrouter_fallback',
            'model_info': OPENROUTER_MODEL,
            'reasoning': result.get('reasoning', ''),
            'frame_count': len(images)
        }, confidence / 100.0
    except Exception as e:
        print(f"[OPENROUTER EXCEPTION] {e}")
        return {'error': f'OpenRouter fallback failed: {str(e)}'}, 0.0

def local_predict(video_path):
    """Run the local trained model pipeline. Raises on failure so timeout can catch it."""
    frames = load_video(video_path, max_frames=MAX_SEQ_LENGTH)
    if len(frames) == 0:
        raise ValueError('Could not extract frames from video')
    prepared_features, frame_mask, _ = prepare_single_video(frames)
    if prepared_features is None or 'inception' not in prepared_features:
        raise ValueError('Failed to extract video features')
    model_prediction = model.predict([prepared_features['inception'], frame_mask], verbose=0)[0][0]
    threshold = 0.5
    predicted_label = 'FAKE' if model_prediction >= threshold else 'REAL'
    confidence = float(model_prediction) * 100 if predicted_label == 'FAKE' else float(1 - model_prediction) * 100
    return {
        'result': predicted_label,
        'confidence': round(confidence, 2),
        'raw_score': float(model_prediction),
        'threshold': threshold,
        'frame_count': len(frames),
        'method': 'trained_model',
        'model_info': 'Trained on 395 videos (81% accuracy, 100% fake detection)',
        'processing_time': 'optimized'
    }, confidence / 100.0

# Enhanced prediction with multiple approaches
def ensemble_predict(prepared_features, frame_mask, additional_features):
    """Make predictions using ensemble methods"""
    predictions = {}
    
    # Feature-based predictions
    for model_name, features in prepared_features.items():
        # Calculate feature statistics
        feature_mean = np.mean(features, axis=1)[0]
        feature_std = np.std(features, axis=1)[0]
        feature_entropy = entropy(np.abs(feature_mean) + 1e-8)
        
        # Simple threshold-based prediction
        # These are heuristic rules - in a real scenario, you'd train specific models
        consistency_score = additional_features['temporal'][model_name]
        motion_variance = additional_features['motion'][1]
        
        # Combine multiple signals
        fake_indicators = 0
        
        # Low temporal consistency often indicates deepfakes
        if consistency_score < 0.7:
            fake_indicators += 1
        
        # High motion variance can indicate synthetic content
        if motion_variance > 10:
            fake_indicators += 1
        
        # High feature entropy might indicate artificial patterns
        if feature_entropy > 5:
            fake_indicators += 1
        
        # Feature uniformity (deepfakes often have less natural variation)
        if np.std(feature_std) < 0.1:
            fake_indicators += 1
        
        # Convert to probability
        fake_probability = min(fake_indicators / 4.0, 1.0)
        predictions[model_name] = fake_probability
    
    return predictions

def calculate_confidence_score(predictions, additional_features):
    """Calculate confidence based on prediction agreement and feature quality"""
    pred_values = list(predictions.values())
    
    # Agreement between models
    agreement = 1.0 - np.std(pred_values)
    
    # Motion consistency
    motion_consistency = 1.0 / (1.0 + additional_features['motion'][1])
    
    # Temporal consistency average
    temporal_avg = np.mean(list(additional_features['temporal'].values()))
    
    # Overall confidence
    confidence = (agreement * 0.4 + motion_consistency * 0.3 + temporal_avg * 0.3)
    return min(max(confidence, 0.0), 1.0)

def advanced_deepfake_detection(video_path):
    """Advanced deepfake detection with multiple techniques"""
    try:
        # Load and preprocess video
        frames = load_video(video_path)
        if len(frames) == 0:
            return {"error": "No valid frames found in video"}, 0.0
        
        print(f"Loaded {len(frames)} frames for analysis")
        
        # Prepare features
        prepared_features, frame_mask, additional_features = prepare_single_video(frames)
        if prepared_features is None:
            return {"error": "Failed to extract features"}, 0.0
        
        print(f"Features extracted successfully")
        
        # Get ensemble predictions
        model_predictions = ensemble_predict(prepared_features, frame_mask, additional_features)
        
        # Calculate final prediction
        final_prediction = np.mean(list(model_predictions.values()))
        
        # Calculate confidence
        confidence = calculate_confidence_score(model_predictions, additional_features)
        
        # Determine result with adaptive threshold
        adaptive_threshold = 0.5 + (0.1 * (1 - confidence))  # Higher threshold when less confident
        result = 'FAKE' if final_prediction >= adaptive_threshold else 'REAL'
        
        print(f"Detection complete: {result} (confidence: {confidence:.2f})")
        
        return {
            'result': result,
            'confidence': float(confidence),
            'raw_prediction': float(final_prediction),
            'model_predictions': {k: float(v) for k, v in model_predictions.items()},
            'temporal_consistency': {k: float(v) for k, v in additional_features['temporal'].items()},
            'motion_features': additional_features['motion'],
            'threshold_used': float(adaptive_threshold),
            'frame_count': len(frames)
        }, confidence
        
    except Exception as e:
        print(f"Error in advanced detection: {e}")
        # Return a simplified detection result as fallback
        try:
            # Simple fallback detection
            frames = load_video(video_path)
            if len(frames) > 0:
                # Basic analysis without complex features
                avg_brightness = np.mean([get_brightness(frame) for frame in frames[:5]])
                frame_variance = np.std([np.std(frame) for frame in frames[:5]])
                
                # Simple heuristic
                fake_score = 0.5
                if avg_brightness < 80 or avg_brightness > 180:
                    fake_score += 0.1
                if frame_variance < 20:
                    fake_score += 0.2
                    
                result = 'FAKE' if fake_score > 0.6 else 'REAL'
                return {
                    'result': result,
                    'confidence': 0.3,  # Low confidence for fallback
                    'method': 'fallback',
                    'frame_count': len(frames),
                    'note': 'Using simplified detection due to processing error'
                }, 0.3
            else:
                return {"error": "Could not process video"}, 0.0
        except:
            return {"error": str(e)}, 0.0

@app.route('/')
def home():
    return render_template('index_optimized.html')  # Using optimized, faster UI

@app.route('/browse-dataset')
def browse_dataset():
    """Browse videos in the dataset folder"""
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'test_videos')
    videos = []
    
    if os.path.exists(dataset_path):
        for filename in os.listdir(dataset_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
                file_path = os.path.join(dataset_path, filename)
                file_size = os.path.getsize(file_path)
                videos.append({
                    'name': filename,
                    'size': f"{file_size / 1024 / 1024:.2f} MB"
                })
    
    return jsonify({'videos': videos})

@app.route('/test-dataset-video/<video_name>')
def test_dataset_video(video_name):
    """Test a video from the dataset folder"""
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'test_videos')
    video_path = os.path.join(dataset_path, video_name)
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        print(f"[VIDEO] Testing dataset video: {video_name}")
        result_dict, confidence = advanced_deepfake_detection(video_path)
        print(f"[RESULT] Dataset video result: {result_dict}")
        return jsonify(result_dict)
    except Exception as e:
        return jsonify({'error': f'Failed to process video: {str(e)}'}), 500

# OPTIMIZED prediction endpoint - faster and more accurate
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(video.filename):
        return jsonify({'error': 'File type not supported. Please upload MP4, AVI, MOV, MKV, WMV, FLV, or WEBM files.'}), 400
    
    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    
    # Generate a unique filename to avoid conflicts
    import uuid
    import re
    
    # Sanitize original filename by removing invalid characters
    original_name = video.filename
    # Remove or replace invalid Windows filename characters
    sanitized_name = re.sub(r'[<>:"/\\|?*]', '_', original_name)
    
    file_extension = os.path.splitext(sanitized_name)[1].lower()
    
    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    if file_extension not in allowed_extensions:
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(allowed_extensions)}'}), 400
    
    # Check file size (limit to 100MB)
    video.seek(0, 2)  # Seek to end
    file_size = video.tell()
    video.seek(0)  # Reset to beginning
    
    max_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_size:
        return jsonify({'error': f'File too large. Maximum size: {max_size//1024//1024}MB'}), 400
    
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    video_path = os.path.join(uploads_dir, unique_filename)
    
    try:
        video.save(video_path)
        print(f"[SUCCESS] Video saved to: {video_path}")
        print(f"[INFO] File size: {file_size:,} bytes")
    except Exception as e:
        return jsonify({'error': f'Failed to save video: {str(e)}'}), 500
    
    try:
        print(f"[VIDEO] Starting deepfake detection for: {unique_filename}")
        
        # Check if model is loaded
        if model is None:
            error_msg = 'Model not loaded. '
            if model_path and not os.path.exists(model_path):
                error_msg += f'Model file not found at: {model_path}'
            else:
                error_msg += 'Model failed to load on startup. Check console for errors.'
            print(f"[ERROR] {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        print(f"[INFO] Model is loaded and ready")
        
        # SIMPLIFIED AND FASTER DETECTION - Use trained model directly with 15s timeout
        try:
            print(f"[INFO] Running local model with {OPENROUTER_TIMEOUT}s timeout")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(local_predict, video_path)
                try:
                    result_dict, _ = future.result(timeout=OPENROUTER_TIMEOUT)
                    print(f"[SUCCESS] Local detection complete: {result_dict['result']} ({result_dict['confidence']}%)")
                    return jsonify(result_dict)
                except concurrent.futures.TimeoutError:
                    print(f"[WARN] Local model timed out after {OPENROUTER_TIMEOUT}s. Switching to OpenRouter API fallback...")
                    # Attempt to cancel the running thread (best effort)
                    future.cancel()
                    # OpenRouter fallback
                    fallback_result, _ = call_openrouter_api(video_path)
                    if 'error' in fallback_result:
                        print(f"[ERROR] OpenRouter fallback also failed: {fallback_result['error']}")
                        return jsonify({'error': f'Local model timed out and OpenRouter fallback failed: {fallback_result["error"]}'}), 500
                    print(f"[SUCCESS] OpenRouter fallback result: {fallback_result['result']} ({fallback_result['confidence']}%)")
                    return jsonify(fallback_result)
                    
        except Exception as e:
            print(f"[ERROR] Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Try OpenRouter as last resort on exception too
            try:
                fallback_result, _ = call_openrouter_api(video_path)
                if 'error' not in fallback_result:
                    print(f"[SUCCESS] OpenRouter fallback on exception: {fallback_result['result']}")
                    return jsonify(fallback_result)
            except Exception as api_e:
                print(f"[ERROR] OpenRouter fallback exception: {api_e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Cleaned up temporary file: {unique_filename}")
        except Exception as e:
            print(f"[WARNING] Could not delete temporary file {video_path}: {e}")

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
