"""
Script to clean up unnecessary files from the workspace
"""
import os
import shutil
from pathlib import Path

def cleanup_workspace():
    """Remove old, duplicate, and unnecessary files"""
    
    base_dir = Path(__file__).parent
    
    # Files to remove
    files_to_remove = [
        # Old/redundant documentation
        "ACCURACY_IMPROVEMENTS.md",
        "APP_RUNNING.md",
        "FIXES_APPLIED.md",
        "INSTALLATION_COMPLETE.md",
        "INTEGRATION_COMPLETE.md",
        "QUICK_REFERENCE.md",
        "READY_TO_TEST.md",
        "SERVER_READY.md",
        "TEST_SCRIPTS_UPDATED.md",
        "TRAINING_COMPLETE.md",
        "TRAINING_PIPELINE_GUIDE.md",
        "TRAINING_SUMMARY.md",
        "UNICODE_FIX_COMPLETE.md",
        
        # Old/redundant test scripts
        "simple_test.py",
        "quick_test.py",
        "test_detection.py",
        "verify_consistency.py",
        
        # Old templates
        "templates/index_old.html",
        
        # Batch files (not needed anymore)
        "install_packages.bat",
        
        # Deployment files (not needed for local dev)
        "Procfile",
        
        # Dataset download script (already have dataset)
        "download_datasets.py",
        
        # Test video (can keep in uploads instead)
        "face_restored_video3.mp4",
        
        # Training image (can regenerate if needed)
        "training_history.png",
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "__pycache__",
        "notebook",  # Jupyter notebook - not needed if using scripts
    ]
    
    removed_count = 0
    skipped_count = 0
    error_count = 0
    
    print("[INFO] Starting workspace cleanup...")
    print("=" * 60)
    
    # Remove files
    print("\n[INFO] Removing unnecessary files...")
    for file_path in files_to_remove:
        full_path = base_dir / file_path
        try:
            if full_path.exists():
                if full_path.is_file():
                    os.remove(full_path)
                    print(f"[REMOVED] {file_path}")
                    removed_count += 1
                else:
                    print(f"[SKIP] {file_path} (not a file)")
                    skipped_count += 1
            else:
                print(f"[SKIP] {file_path} (not found)")
                skipped_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to remove {file_path}: {e}")
            error_count += 1
    
    # Remove directories
    print("\n[INFO] Removing unnecessary directories...")
    for dir_path in dirs_to_remove:
        full_path = base_dir / dir_path
        try:
            if full_path.exists():
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                    print(f"[REMOVED] {dir_path}/")
                    removed_count += 1
                else:
                    print(f"[SKIP] {dir_path} (not a directory)")
                    skipped_count += 1
            else:
                print(f"[SKIP] {dir_path}/ (not found)")
                skipped_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to remove {dir_path}: {e}")
            error_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY] Cleanup complete!")
    print(f"  - Files/Folders removed: {removed_count}")
    print(f"  - Skipped (not found): {skipped_count}")
    print(f"  - Errors: {error_count}")
    print("=" * 60)
    
    # Show remaining important files
    print("\n[INFO] Remaining important files:")
    important_files = [
        "app.py",
        "start_server.py",
        "train_model.py",
        "train_pipeline.py",
        "test_new_model.py",
        "quick_test_video.py",
        "organize_videos.py",
        "requirements.txt",
        "README.md",
        "model/deepfake_video_model.h5",
        "templates/index.html",
        "templates/index_optimized.html",
    ]
    
    for file_path in important_files:
        full_path = base_dir / file_path
        if full_path.exists():
            if full_path.is_file():
                size = full_path.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"  [KEEP] {file_path} ({size_mb:.2f} MB)")
            else:
                print(f"  [KEEP] {file_path} (directory)")
    
    print("\n[SUCCESS] Workspace cleaned successfully!")
    print("[NOTE] You can now delete this cleanup script if you want.")

if __name__ == "__main__":
    try:
        # Ask for confirmation
        print("=" * 60)
        print("WORKSPACE CLEANUP SCRIPT")
        print("=" * 60)
        print("\nThis will remove:")
        print("  - Old documentation files (.md)")
        print("  - Redundant test scripts")
        print("  - Old template files")
        print("  - Jupyter notebooks")
        print("  - __pycache__ directories")
        print("  - Deployment files (Procfile)")
        print("  - Dataset download script")
        print("\nImportant files will be kept:")
        print("  - app.py, start_server.py")
        print("  - Training scripts (train_model.py, train_pipeline.py)")
        print("  - Test scripts (test_new_model.py, quick_test_video.py)")
        print("  - Model file (deepfake_video_model.h5)")
        print("  - Templates (index.html, index_optimized.html)")
        print("  - Dataset videos")
        print("\n" + "=" * 60)
        
        response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print("\n")
            cleanup_workspace()
        else:
            print("\n[CANCELLED] Cleanup cancelled by user.")
    
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Cleanup cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
