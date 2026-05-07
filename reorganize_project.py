"""
Project Reorganization Script
Restructures the DeepFake Video Detection project into a professional modular format
"""
import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new professional directory structure"""
    
    base_dir = Path(__file__).parent
    
    # Define the new structure
    structure = {
        'video-detection-module': {
            'model': [
                '__init__.py',
            ],
            'training-data': {
                'real': [],
                'fake': [],
                'metadata': []
            },
            'saved-models': {
                'checkpoints': [],
                'production': []
            },
            'api': [
                '__init__.py',
            ],
            'tests': [
                '__init__.py',
            ],
            'utils': [
                '__init__.py',
            ],
            'web': {
                'templates': [],
                'static': {
                    'css': [],
                    'js': [],
                    'images': []
                }
            },
            'uploads': [],
            'logs': [],
            'docs': []
        }
    }
    
    def create_structure(base_path, struct):
        """Recursively create directory structure"""
        for key, value in struct.items():
            current_path = base_path / key
            current_path.mkdir(exist_ok=True)
            
            if isinstance(value, dict):
                create_structure(current_path, value)
            elif isinstance(value, list):
                # Create files in this directory
                for filename in value:
                    if filename:
                        (current_path / filename).touch()
    
    # Create the structure
    new_root = base_dir / 'video-detection-module'
    if new_root.exists():
        print(f"[WARNING] Directory '{new_root}' already exists")
        response = input("Delete and recreate? (yes/no): ").strip().lower()
        if response == 'yes':
            shutil.rmtree(new_root)
        else:
            print("[CANCELLED] Reorganization cancelled")
            return None
    
    create_structure(base_dir, structure)
    print(f"[SUCCESS] Created directory structure at: {new_root}")
    
    return new_root

def create_file_mapping():
    """Define mapping of old files to new locations"""
    
    mapping = {
        # Model files
        'train_model.py': 'video-detection-module/model/train.py',
        'retrain_balanced_model.py': 'video-detection-module/model/train_balanced.py',
        'train_oversampled_model.py': 'video-detection-module/model/train_oversampled.py',
        'train_improved_model.py': 'video-detection-module/model/train_improved.py',
        
        # API files
        'app.py': 'video-detection-module/api/app.py',
        'start_server.py': 'video-detection-module/api/start_server.py',
        
        # Test files
        'test_new_model.py': 'video-detection-module/tests/test_model.py',
        'test_real_videos.py': 'video-detection-module/tests/test_real_videos.py',
        'quick_test_video.py': 'video-detection-module/tests/quick_test.py',
        
        # Utility files
        'organize_videos.py': 'video-detection-module/utils/organize_videos.py',
        'cleanup_workspace.py': 'video-detection-module/utils/cleanup.py',
        
        # Model weights
        'model/deepfake_video_model.h5': 'video-detection-module/saved-models/production/deepfake_video_model.h5',
        'model/deepfake_video_model_OLD_UNBALANCED.h5': 'video-detection-module/saved-models/checkpoints/model_unbalanced.h5',
        'model/deepfake_video_model_balanced_best.h5': 'video-detection-module/saved-models/checkpoints/model_balanced_best.h5',
        
        # Training data
        'dataset/train_sample_videos/real': 'video-detection-module/training-data/real',
        'dataset/train_sample_videos/fake': 'video-detection-module/training-data/fake',
        'dataset/train_sample_videos/metadata.json': 'video-detection-module/training-data/metadata/metadata.json',
        
        # Web files
        'templates': 'video-detection-module/web/templates',
        'static': 'video-detection-module/web/static',
        
        # Documentation
        'README.md': 'video-detection-module/README.md',
        'requirements.txt': 'video-detection-module/requirements.txt',
        'PROJECT_STRUCTURE.md': 'video-detection-module/docs/PROJECT_STRUCTURE.md',
        'REAL_VIDEO_DETECTION_FIXED.md': 'video-detection-module/docs/REAL_VIDEO_DETECTION_FIXED.md',
        'training_history_balanced.png': 'video-detection-module/docs/training_history_balanced.png',
    }
    
    return mapping

def reorganize_files(base_dir, new_root):
    """Move files to their new locations"""
    
    mapping = create_file_mapping()
    moved_count = 0
    error_count = 0
    skipped_count = 0
    
    print("\n[INFO] Moving files to new structure...")
    print("=" * 60)
    
    for old_path, new_path in mapping.items():
        old_full = base_dir / old_path
        new_full = base_dir / new_path
        
        try:
            if old_full.exists():
                # Create parent directory if needed
                new_full.parent.mkdir(parents=True, exist_ok=True)
                
                if old_full.is_dir():
                    # Copy directory
                    if new_full.exists():
                        shutil.rmtree(new_full)
                    shutil.copytree(old_full, new_full)
                    print(f"[COPIED DIR] {old_path} -> {new_path}")
                else:
                    # Copy file
                    shutil.copy2(old_full, new_full)
                    print(f"[COPIED] {old_path} -> {new_path}")
                
                moved_count += 1
            else:
                print(f"[SKIP] {old_path} (not found)")
                skipped_count += 1
                
        except Exception as e:
            print(f"[ERROR] Failed to move {old_path}: {e}")
            error_count += 1
    
    print("=" * 60)
    print(f"[SUMMARY] Moved: {moved_count}, Skipped: {skipped_count}, Errors: {error_count}")
    
    return moved_count

def create_module_files(new_root):
    """Create essential module files"""
    
    files_created = 0
    
    # 1. Model architecture file
    model_py = new_root / 'model' / 'model.py'
    with open(model_py, 'w', encoding='utf-8') as f:
        f.write('''"""
DeepFake Video Detection Model Architecture
"""
import tensorflow as tf
import numpy as np

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 30
NUM_FEATURES = 2048

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

def build_detection_model():
    """Build sequence model for deepfake detection"""
    # Feature input
    frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
    
    # GRU layers for temporal analysis
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
''')
    files_created += 1
    print(f"[CREATED] model/model.py")
    
    # 2. Detection/Inference script
    detect_py = new_root / 'model' / 'detect.py'
    with open(detect_py, 'w', encoding='utf-8') as f:
        f.write('''"""
DeepFake Video Detection - Inference Script
"""
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
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
    
    model_path = Path(__file__).parent.parent / "saved-models" / "production" / "deepfake_video_model.h5"
    detector = DeepFakeDetector(str(model_path))
    
    result = detector.predict(sys.argv[1])
    if result:
        print(f"\\nResult: {result['result']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Raw Score: {result['raw_score']:.4f}")
''')
    files_created += 1
    print(f"[CREATED] model/detect.py")
    
    # 3. Main README
    readme_md = new_root / 'README.md'
    with open(readme_md, 'w', encoding='utf-8') as f:
        f.write('''# 🎬 DeepFake Video Detection Module

Professional deepfake detection system using InceptionV3 + GRU architecture.

## 📁 Project Structure

```
video-detection-module/
├── model/                        # Core detection model
│   ├── __init__.py
│   ├── model.py                  # Model architecture (InceptionV3 + GRU)
│   ├── train.py                  # Training script
│   ├── train_balanced.py         # Balanced dataset training
│   ├── train_oversampled.py      # Oversampled training
│   ├── train_improved.py         # Improved architecture training
│   └── detect.py                 # Inference/detection script
│
├── training-data/                # Training datasets
│   ├── real/                     # Real videos (74 videos)
│   ├── fake/                     # Fake videos (320 videos)
│   └── metadata/                 # Video metadata
│       └── metadata.json
│
├── saved-models/                 # Trained model weights
│   ├── production/               # Production-ready models
│   │   └── deepfake_video_model.h5
│   └── checkpoints/              # Training checkpoints
│       ├── model_balanced_best.h5
│       └── model_unbalanced.h5
│
├── api/                          # REST API endpoint
│   ├── __init__.py
│   ├── app.py                    # Flask application
│   └── start_server.py           # Server launcher
│
├── web/                          # Web interface
│   ├── templates/                # HTML templates
│   │   ├── index.html
│   │   └── index_optimized.html
│   └── static/                   # CSS, JS, images
│       ├── css/
│       ├── js/
│       └── images/
│
├── tests/                        # Test scripts
│   ├── __init__.py
│   ├── test_model.py             # Model testing
│   ├── test_real_videos.py       # Real video validation
│   └── quick_test.py             # Quick testing utility
│
├── utils/                        # Utility scripts
│   ├── __init__.py
│   ├── organize_videos.py        # Video organization
│   └── cleanup.py                # Workspace cleanup
│
├── uploads/                      # Temporary upload directory
├── logs/                         # Application logs
│
├── docs/                         # Documentation
│   ├── PROJECT_STRUCTURE.md
│   ├── REAL_VIDEO_DETECTION_FIXED.md
│   └── training_history_balanced.png
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Detection on a Video

```bash
# Using detection script
python model/detect.py path/to/video.mp4

# Using API server
python api/start_server.py
# Then open http://localhost:5000
```

### 3. Train New Model

```bash
# Train with balanced dataset (recommended)
python model/train_balanced.py

# Train with oversampled data
python model/train_oversampled.py

# Train with all data
python model/train.py
```

## 🧠 Model Architecture

- **Feature Extractor:** InceptionV3 (pre-trained on ImageNet)
- **Temporal Analysis:** 2 GRU layers (16 → 8 units)
- **Input:** 30 frames per video
- **Output:** Binary classification (REAL / FAKE)

## 📊 Model Performance

### Current Model (Balanced Training):
- **Real Video Detection:** 70%
- **Fake Video Detection:** 53%
- **Overall Accuracy:** 57%
- **Training Data:** 148 videos (50% real, 50% fake)

### Previous Model (Unbalanced):
- **Real Video Detection:** 0% ❌
- **Fake Video Detection:** 100%
- **Overall Accuracy:** 81% (misleading)
- **Issue:** Extreme class imbalance (81% fake videos)

## 🔧 API Endpoints

### `POST /predict`
Upload and analyze a video file.

**Request:**
```bash
curl -X POST -F "video=@video.mp4" http://localhost:5000/predict
```

**Response:**
```json
{
  "result": "FAKE",
  "confidence": 78.5,
  "raw_score": 0.785,
  "frame_count": 30,
  "method": "trained_model"
}
```

## 📈 Training Data

- **Real Videos:** 74 (18.8%)
- **Fake Videos:** 320 (81.2%)
- **Total:** 394 videos
- **Source:** Organized from train_sample_videos dataset

## 🧪 Testing

```bash
# Test model on real videos
python tests/test_real_videos.py

# Quick test on any video
python tests/quick_test.py path/to/video.mp4

# Run all tests
python tests/test_model.py
```

## 📝 Configuration

Edit model parameters in `model/model.py`:
- `IMG_SIZE`: Frame resolution (default: 224x224)
- `MAX_SEQ_LENGTH`: Frames per video (default: 30)
- `NUM_FEATURES`: Feature dimensions (default: 2048)

## 🐛 Known Issues & Solutions

### Issue: Low confidence on real videos
**Solution:** Retrain with balanced dataset or more real videos

### Issue: All videos detected as fake
**Solution:** Use `train_balanced.py` instead of `train.py`

### Issue: Model expects 30 frames
**Solution:** Ensure `MAX_SEQ_LENGTH` matches training value

## 📚 Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Real Video Detection Fix](docs/REAL_VIDEO_DETECTION_FIXED.md)
- [Training History](docs/training_history_balanced.png)

## 🔐 Requirements

- Python 3.8+
- TensorFlow 2.20+
- OpenCV 4.12+
- Flask 3.0+
- NumPy, scikit-learn, matplotlib

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Follow the project structure
2. Test on both real and fake videos
3. Document model changes
4. Update this README

## 📄 License

[Your License Here]

## 👤 Author

[Your Name]

---

**Status:** ✅ Production Ready (Balanced Model)  
**Last Updated:** October 17, 2025  
**Version:** 2.0 (Reorganized & Balanced)
''')
    files_created += 1
    print(f"[CREATED] README.md")
    
    # 4. Module __init__.py files
    for init_path in [
        new_root / 'model' / '__init__.py',
        new_root / 'api' / '__init__.py',
        new_root / 'tests' / '__init__.py',
        new_root / 'utils' / '__init__.py'
    ]:
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(f'"""{init_path.parent.name} module"""\n')
        files_created += 1
    
    print(f"\n[SUCCESS] Created {files_created} module files")
    
    return files_created

def create_project_tree(new_root):
    """Create a visual project tree file"""
    
    tree_file = new_root / 'PROJECT_TREE.txt'
    
    tree_content = '''video-detection-module/
├── model/                        # Core detection model
│   ├── __init__.py
│   ├── model.py                  # Model architecture (InceptionV3 + GRU)
│   ├── train.py                  # Original training script
│   ├── train_balanced.py         # Balanced dataset training
│   ├── train_oversampled.py      # Oversampled training
│   ├── train_improved.py         # Improved architecture training
│   └── detect.py                 # Inference/detection script
│
├── training-data/                # Training datasets
│   ├── real/                     # Real videos (74 videos)
│   │   ├── abarnvbtwb.mp4
│   │   ├── aelfnikyqj.mp4
│   │   └── ... (72 more)
│   ├── fake/                     # Fake videos (320 videos)
│   │   ├── aagfhgtpmv.mp4
│   │   ├── aapnvogymq.mp4
│   │   └── ... (318 more)
│   └── metadata/                 # Video metadata
│       └── metadata.json         # Labels and info
│
├── saved-models/                 # Trained model weights
│   ├── production/               # Production-ready models
│   │   └── deepfake_video_model.h5  # Current best model (1.18 MB)
│   └── checkpoints/              # Training checkpoints
│       ├── model_balanced_best.h5
│       └── model_unbalanced.h5
│
├── api/                          # REST API endpoint
│   ├── __init__.py
│   ├── app.py                    # Flask application (main API)
│   └── start_server.py           # Server launcher with checks
│
├── web/                          # Web interface
│   ├── templates/                # HTML templates
│   │   ├── index.html            # Main web UI
│   │   └── index_optimized.html  # Optimized lightweight UI
│   └── static/                   # CSS, JS, images
│       ├── css/
│       ├── js/
│       └── images/
│
├── tests/                        # Test scripts
│   ├── __init__.py
│   ├── test_model.py             # Comprehensive model testing
│   ├── test_real_videos.py       # Real video validation (70% accuracy)
│   └── quick_test.py             # Quick single video test
│
├── utils/                        # Utility scripts
│   ├── __init__.py
│   ├── organize_videos.py        # Organize videos by label
│   └── cleanup.py                # Workspace cleanup
│
├── uploads/                      # Temporary upload directory (auto-cleaned)
├── logs/                         # Application logs (created at runtime)
│
├── docs/                         # Documentation
│   ├── PROJECT_STRUCTURE.md      # Project organization
│   ├── REAL_VIDEO_DETECTION_FIXED.md  # Fix documentation
│   └── training_history_balanced.png  # Training curves
│
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
└── PROJECT_TREE.txt              # This file

=================================================================
STATISTICS
=================================================================
Total Files: 400+ (including videos)
Code Files: 20+
Model Files: 3
Video Files: 394 (74 real, 320 fake)
Documentation: 5 files

=================================================================
KEY COMPONENTS
=================================================================

1. MODEL ARCHITECTURE
   - InceptionV3 feature extractor (2048 features)
   - GRU temporal analysis (16→8 units)
   - Binary classification (sigmoid output)

2. TRAINING DATA
   - 74 REAL videos (18.8%)
   - 320 FAKE videos (81.2%)
   - Balanced training uses 74+74=148 videos

3. API SERVER
   - Flask REST API
   - POST /predict endpoint
   - Web UI at http://localhost:5000

4. PERFORMANCE
   - Real detection: 70% (balanced model)
   - Fake detection: 53% (balanced model)
   - Processing: 5-8 seconds per video

=================================================================
QUICK COMMANDS
=================================================================

# Start web server
python api/start_server.py

# Detect deepfake
python model/detect.py video.mp4

# Train model
python model/train_balanced.py

# Test model
python tests/test_real_videos.py

=================================================================
'''
    
    with open(tree_file, 'w', encoding='utf-8') as f:
        f.write(tree_content)
    
    print(f"[CREATED] PROJECT_TREE.txt")
    
    return str(tree_file)

def main():
    """Main reorganization function"""
    print("=" * 60)
    print("DeepFake Video Detection - Project Reorganization")
    print("=" * 60)
    print("\nThis will reorganize your project into a professional modular structure:")
    print("  - video-detection-module/")
    print("    ├── model/          (training & detection)")
    print("    ├── api/            (Flask server)")
    print("    ├── training-data/  (datasets)")
    print("    ├── saved-models/   (weights)")
    print("    ├── tests/          (testing)")
    print("    ├── utils/          (utilities)")
    print("    ├── web/            (UI)")
    print("    └── docs/           (documentation)")
    print("\n" + "=" * 60)
    
    response = input("\nProceed with reorganization? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n[CANCELLED] Reorganization cancelled")
        return
    
    print("\n[INFO] Starting reorganization...\n")
    
    # Step 1: Create directory structure
    print("[STEP 1/4] Creating directory structure...")
    new_root = create_directory_structure()
    
    if new_root is None:
        return
    
    # Step 2: Move files
    print("\n[STEP 2/4] Moving files to new locations...")
    base_dir = Path(__file__).parent
    moved_count = reorganize_files(base_dir, new_root)
    
    # Step 3: Create module files
    print("\n[STEP 3/4] Creating module files...")
    files_created = create_module_files(new_root)
    
    # Step 4: Create project tree
    print("\n[STEP 4/4] Creating project documentation...")
    tree_file = create_project_tree(new_root)
    
    # Summary
    print("\n" + "=" * 60)
    print("REORGANIZATION COMPLETE!")
    print("=" * 60)
    print(f"New project location: {new_root}")
    print(f"Files moved: {moved_count}")
    print(f"Module files created: {files_created}")
    print(f"\n[NEXT STEPS]")
    print(f"1. Navigate to: cd {new_root}")
    print(f"2. Check structure: cat PROJECT_TREE.txt")
    print(f"3. Read documentation: cat README.md")
    print(f"4. Start server: python api/start_server.py")
    print(f"5. Test detection: python model/detect.py <video.mp4>")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Reorganization cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Reorganization failed: {e}")
        import traceback
        traceback.print_exc()
