"""
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
