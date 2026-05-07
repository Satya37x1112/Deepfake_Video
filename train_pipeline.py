#!/usr/bin/env python3
"""
Integrated Training Pipeline for Deepfake Detection
Reuses feature extraction from test_new_model.py for consistency
Supports full training, fine-tuning, and comprehensive evaluation
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# SHARED PREPROCESSING (Same as test_new_model.py for consistency)
# ============================================================================

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
    """Load and preprocess video - SAME as test_new_model.py"""
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
    """Build InceptionV3 feature extractor - SAME as test_new_model.py"""
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

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TrainingProgress(tf.keras.callbacks.Callback):
    """Custom callback for detailed progress tracking"""
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{self.total_epochs}")
        print(f"{'='*70}")
    
    def on_epoch_end(self, epoch, logs=None):
        duration = (datetime.now() - self.epoch_start_time).total_seconds()
        
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   ⏱️  Duration: {duration:.1f}s")
        print(f"   📈 Training   - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
        print(f"   📉 Validation - Loss: {logs['val_loss']:.4f}, Accuracy: {logs['val_accuracy']:.4f}")
        
        if 'lr' in logs:
            print(f"   🎯 Learning Rate: {logs['lr']:.6f}")

def load_dataset(new_video_path=None, new_video_label="FAKE", metadata_path=None):
    """
    Load dataset with optional new video
    
    Args:
        new_video_path: Path to additional video to include
        new_video_label: Label for new video ("FAKE" or "REAL")
        metadata_path: Path to metadata.json (auto-detected if None)
    
    Returns:
        video_paths: List of video file paths
        labels: List of labels (0=REAL, 1=FAKE)
        video_names: List of video filenames
    """
    video_paths = []
    labels = []
    video_names = []
    
    # Add new video if provided
    if new_video_path and os.path.exists(new_video_path):
        print(f"📹 Adding new video: {Path(new_video_path).name}")
        video_paths.append(new_video_path)
        labels.append(1 if new_video_label == "FAKE" else 0)
        video_names.append(Path(new_video_path).name)
        print(f"   Label: {new_video_label}")
    
    # Load existing dataset
    if metadata_path is None:
        metadata_path = Path("dataset/train_sample_videos/metadata.json")
    
    if metadata_path.exists():
        video_folder = metadata_path.parent
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📂 Loading dataset from: {metadata_path}")
        for filename, info in metadata.items():
            video_path = video_folder / filename
            if video_path.exists():
                video_paths.append(str(video_path))
                labels.append(1 if info['label'] == 'FAKE' else 0)
                video_names.append(filename)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total videos: {len(video_paths)}")
    print(f"   FAKE videos: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"   REAL videos: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    return video_paths, labels, video_names

def extract_features_batch(video_paths, labels, feature_extractor, batch_size=10):
    """
    Extract features from videos with progress tracking
    
    Args:
        video_paths: List of video file paths
        labels: List of labels
        feature_extractor: TensorFlow model for feature extraction
        batch_size: Number of videos to process before showing progress
    
    Returns:
        features: Array of shape (n_videos, MAX_SEQ_LENGTH, NUM_FEATURES)
        masks: Array of shape (n_videos, MAX_SEQ_LENGTH)
        labels: Array of labels
    """
    print(f"\n🎬 Extracting features from {len(video_paths)} videos...")
    print(f"   Using InceptionV3 feature extractor")
    print(f"   Progress updates every {batch_size} videos\n")
    
    all_features = []
    all_masks = []
    all_labels = []
    
    total = len(video_paths)
    
    for idx, (video_path, label) in enumerate(zip(video_paths, labels), 1):
        try:
            # Load video
            frames = load_video(video_path)
            
            if len(frames) == 0:
                print(f"⚠️  Skipping {Path(video_path).name} - no frames loaded")
                continue
            
            # Extract features
            features = []
            for frame in frames:
                feature = feature_extractor.predict(frame[None, ...], verbose=0)
                features.append(feature[0])
            
            features = np.array(features)
            
            # Create feature array with padding
            feature_array = np.zeros(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
            frame_mask = np.zeros(shape=(MAX_SEQ_LENGTH,), dtype="bool")
            
            length = min(MAX_SEQ_LENGTH, len(features))
            feature_array[:length, :] = features[:length]
            frame_mask[:length] = 1  # 1 = not masked
            
            all_features.append(feature_array)
            all_masks.append(frame_mask)
            all_labels.append(label)
            
            # Progress update
            if idx % batch_size == 0:
                progress = idx / total * 100
                print(f"   ✓ Processed {idx}/{total} videos ({progress:.1f}%)")
                
        except Exception as e:
            print(f"❌ Error processing {Path(video_path).name}: {e}")
            continue
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Successfully processed: {len(all_features)}/{total} videos")
    
    return (
        np.array(all_features),
        np.array(all_masks),
        np.array(all_labels)
    )

def build_model(num_features=NUM_FEATURES, seq_length=MAX_SEQ_LENGTH):
    """
    Build GRU-based sequence model for deepfake detection
    Architecture matches the model structure from train_model.py
    """
    # Feature input
    frame_features_input = tf.keras.Input((seq_length, num_features), name='features_input')
    mask_input = tf.keras.Input((seq_length,), dtype="bool", name='mask_input')
    
    # GRU layers for temporal analysis
    x = tf.keras.layers.GRU(16, return_sequences=True, name='gru_1')(
        frame_features_input, mask=mask_input
    )
    x = tf.keras.layers.GRU(8, name='gru_2')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout')(x)
    x = tf.keras.layers.Dense(8, activation="relu", name='dense_hidden')(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name='output')(x)
    
    model = tf.keras.Model([frame_features_input, mask_input], output, name='deepfake_detector')
    
    return model

def compile_model(model, learning_rate=0.001):
    """Compile model with optimizer and metrics"""
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

def plot_training_history(history, save_path='training_history_detailed.png'):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', marker='s')
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision & Recall
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision', marker='o')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision', marker='s')
        axes[1, 0].plot(history.history['recall'], label='Train Recall', marker='^')
        axes[1, 0].plot(history.history['val_recall'], label='Val Recall', marker='v')
        axes[1, 0].set_title('Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    if 'auc' in history.history:
        axes[1, 1].plot(history.history['auc'], label='Train AUC', marker='o')
        axes[1, 1].plot(history.history['val_auc'], label='Val AUC', marker='s')
        axes[1, 1].set_title('AUC Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training history saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['REAL', 'FAKE'],
        yticklabels=['REAL', 'FAKE'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path='roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 ROC curve saved to: {save_path}")
    plt.close()

def evaluate_model(model, features, masks, labels, threshold=0.5):
    """
    Comprehensive model evaluation
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{'='*70}")
    print("📊 MODEL EVALUATION")
    print(f"{'='*70}")
    
    # Get predictions
    predictions_proba = model.predict([features, masks], verbose=0).flatten()
    predictions_binary = (predictions_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions_binary)
    precision = precision_score(labels, predictions_binary, zero_division=0)
    recall = recall_score(labels, predictions_binary, zero_division=0)
    f1 = f1_score(labels, predictions_binary, zero_division=0)
    auc = roc_auc_score(labels, predictions_proba)
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(
        labels, 
        predictions_binary,
        target_names=['REAL', 'FAKE'],
        digits=4
    ))
    
    # Print confusion matrix
    cm = confusion_matrix(labels, predictions_binary)
    print("\n🎯 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              REAL    FAKE")
    print(f"Actual REAL   {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"       FAKE   {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    # Print summary metrics
    print(f"\n📈 Summary Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")
    
    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Fake Detection Rate (same as recall)
    fake_detection_rate = recall
    
    print(f"\n🎯 Additional Metrics:")
    print(f"   Specificity (REAL detection): {specificity:.4f}")
    print(f"   Fake Detection Rate:          {fake_detection_rate:.4f}")
    print(f"   False Positive Rate:          {fp/(fp+tn) if (fp+tn) > 0 else 0:.4f}")
    print(f"   False Negative Rate:          {fn/(fn+tp) if (fn+tp) > 0 else 0:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': cm.tolist(),
        'predictions_proba': predictions_proba,
        'predictions_binary': predictions_binary
    }
    
    return metrics

def train_model(
    model,
    train_features,
    train_masks,
    train_labels,
    val_features,
    val_masks,
    val_labels,
    epochs=50,
    batch_size=8,
    initial_epoch=0
):
    """
    Train model with comprehensive callbacks
    
    Args:
        model: Compiled Keras model
        train_features, train_masks, train_labels: Training data
        val_features, val_masks, val_labels: Validation data
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        initial_epoch: Starting epoch (for fine-tuning)
    
    Returns:
        history: Training history object
    """
    print(f"\n{'='*70}")
    print("🚀 TRAINING MODEL")
    print(f"{'='*70}")
    print(f"Training samples:   {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    print(f"Epochs:             {epochs}")
    print(f"Batch size:         {batch_size}")
    print(f"Initial epoch:      {initial_epoch}")
    
    # Create model directory
    os.makedirs('./model', exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Progress tracking
        TrainingProgress(epochs),
        
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./model/deepfake_video_model_best.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Train model
    history = model.fit(
        [train_features, train_masks],
        train_labels,
        validation_data=([val_features, val_masks], val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=initial_epoch
    )
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    
    return history

def save_training_report(history, metrics, config, save_path='training_report.txt'):
    """Save comprehensive training report"""
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DEEPFAKE DETECTION MODEL - TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("TRAINING HISTORY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Epochs: {len(history.history['loss'])}\n")
        f.write(f"Best Val Accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"Best Val Loss: {min(history.history['val_loss']):.4f}\n")
        f.write(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write("\n")
        
        f.write("EVALUATION METRICS\n")
        f.write("-"*70 + "\n")
        for key, value in metrics.items():
            if key not in ['predictions_proba', 'predictions_binary', 'confusion_matrix']:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-"*70 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"                Predicted\n")
        f.write(f"              REAL    FAKE\n")
        f.write(f"Actual REAL   {cm[0][0]:4d}   {cm[0][1]:4d}\n")
        f.write(f"       FAKE   {cm[1][0]:4d}   {cm[1][1]:4d}\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
    
    print(f"\n📄 Training report saved to: {save_path}")

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🎬 DEEPFAKE DETECTION - INTEGRATED TRAINING PIPELINE")
    print("="*70)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    config = {
        'new_video_path': r"C:\Users\manoh\OneDrive\Desktop\DeepFakeVideoDetection\DeepFakeVideoDetection\face_restored_video3.mp4",
        'new_video_label': "FAKE",
        'model_path': './model/deepfake_video_model.h5',
        'epochs': 50,
        'batch_size': 8,
        'learning_rate': 0.001,
        'test_size': 0.2,
        'random_state': 42,
        'threshold': 0.5
    }
    
    # Check for existing model
    model_exists = os.path.exists(config['model_path'])
    
    if model_exists:
        print(f"\n⚠️  Existing model found: {config['model_path']}")
        print("\nOptions:")
        print("1. Full retraining (start from scratch)")
        print("2. Fine-tuning (continue training existing model)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '3':
            print("👋 Exiting...")
            return
        elif choice == '2':
            mode = 'fine-tune'
            print("\n🔧 Fine-tuning mode selected")
        else:
            mode = 'full-train'
            print("\n🆕 Full retraining mode selected")
    else:
        mode = 'full-train'
        print("\n🆕 No existing model found. Starting fresh training.")
    
    # ========================================================================
    # LOAD DATASET
    # ========================================================================
    
    video_paths, labels, video_names = load_dataset(
        new_video_path=config['new_video_path'],
        new_video_label=config['new_video_label']
    )
    
    if len(video_paths) == 0:
        print("❌ No videos found! Exiting...")
        return
    
    # ========================================================================
    # EXTRACT FEATURES
    # ========================================================================
    
    print("\n🔧 Building feature extractor...")
    feature_extractor = build_feature_extractor()
    print("✅ Feature extractor ready!")
    
    features, masks, labels = extract_features_batch(
        video_paths, 
        labels, 
        feature_extractor,
        batch_size=10
    )
    
    if len(features) == 0:
        print("❌ No features extracted! Exiting...")
        return
    
    print(f"\n✅ Feature extraction successful!")
    print(f"   Features shape: {features.shape}")
    print(f"   Masks shape: {masks.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # ========================================================================
    # SPLIT DATA
    # ========================================================================
    
    train_features, val_features, train_masks, val_masks, train_labels, val_labels = train_test_split(
        features, masks, labels,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=labels
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training set: {len(train_features)} samples")
    print(f"   Validation set: {len(val_features)} samples")
    print(f"   Train FAKE: {sum(train_labels)}/{len(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
    print(f"   Val FAKE: {sum(val_labels)}/{len(val_labels)} ({sum(val_labels)/len(val_labels)*100:.1f}%)")
    
    # ========================================================================
    # BUILD/LOAD MODEL
    # ========================================================================
    
    if mode == 'fine-tune':
        print(f"\n🔧 Loading existing model for fine-tuning...")
        try:
            model = tf.keras.models.load_model(config['model_path'])
            print("✅ Model loaded successfully!")
            
            # Optionally reduce learning rate for fine-tuning
            fine_tune_lr = config['learning_rate'] * 0.1
            model = compile_model(model, learning_rate=fine_tune_lr)
            print(f"   Fine-tuning learning rate: {fine_tune_lr}")
            
            initial_epoch = 0  # Can be adjusted if tracking epochs
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Switching to full retraining...")
            mode = 'full-train'
    
    if mode == 'full-train':
        print("\n🔧 Building new model...")
        model = build_model()
        model = compile_model(model, learning_rate=config['learning_rate'])
        initial_epoch = 0
        
        print("\n📋 Model Architecture:")
        model.summary()
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    
    history = train_model(
        model,
        train_features,
        train_masks,
        train_labels,
        val_features,
        val_masks,
        val_labels,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        initial_epoch=initial_epoch
    )
    
    # ========================================================================
    # EVALUATE MODEL
    # ========================================================================
    
    metrics = evaluate_model(
        model,
        val_features,
        val_masks,
        val_labels,
        threshold=config['threshold']
    )
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    
    print(f"\n💾 Saving model...")
    model.save(config['model_path'])
    print(f"✅ Model saved to: {config['model_path']}")
    
    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    
    print(f"\n📊 Generating visualizations...")
    
    plot_training_history(history)
    plot_confusion_matrix(metrics['predictions_binary'], val_labels)
    plot_roc_curve(val_labels, metrics['predictions_proba'])
    
    # ========================================================================
    # SAVE REPORT
    # ========================================================================
    
    save_training_report(history, metrics, config)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\n✅ Model saved: {config['model_path']}")
    print(f"✅ Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"✅ Final metrics: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['auc']:.4f}")
    print(f"\n📊 Generated files:")
    print(f"   - training_history_detailed.png")
    print(f"   - confusion_matrix.png")
    print(f"   - roc_curve.png")
    print(f"   - training_report.txt")
    print(f"\n🧪 To test the model, run: python test_new_model.py")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
