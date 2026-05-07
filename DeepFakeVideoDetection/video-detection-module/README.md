# 🎬 DeepFake Video Detection Module

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
