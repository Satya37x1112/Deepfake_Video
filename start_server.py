"""
Quick Start Script for DeepFake Detection
This will check all dependencies and start the server
"""
import os
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 70)
print("[STARTUP] DeepFake Detection System")
print("=" * 70)

# Check Python version
print(f"\n[OK] Python Version: {sys.version.split()[0]}")

# Check if we're in the right directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"[OK] Working Directory: {current_dir}")

# Check model file
model_path = os.path.join(current_dir, 'model', 'deepfake_video_model.h5')
if os.path.exists(model_path):
    model_size = os.path.getsize(model_path)
    print(f"[SUCCESS] Model Found: {model_size:,} bytes")
else:
    print(f"[ERROR] Model NOT found at: {model_path}")
    print("   Please ensure deepfake_video_model.h5 exists in the model folder")
    sys.exit(1)

# Check required modules
print("\n[CHECK] Checking Dependencies...")
required_modules = {
    'flask': 'Flask',
    'tensorflow': 'TensorFlow',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'sklearn': 'Scikit-learn'
}

all_ok = True
for module_name, display_name in required_modules.items():
    try:
        __import__(module_name)
        print(f"  [OK] {display_name}")
    except ImportError:
        print(f"  [ERROR] {display_name} NOT INSTALLED")
        all_ok = False

if not all_ok:
    print("\n[ERROR] Some dependencies are missing. Run: pip install -r requirements.txt")
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] All checks passed! Starting Flask server...")
print("=" * 70)
print("\n[INFO] Server will be available at: http://localhost:3000")
print("[INFO] Open this URL in your browser to use the app")
print("\n[WARNING] Press CTRL+C to stop the server\n")

# Import and run the app
try:
    from app import app
    app.run(debug=True, host='0.0.0.0', port=3000)
except KeyboardInterrupt:
    print("\n\n[STOP] Server stopped by user")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
