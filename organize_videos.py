"""
Script to organize videos into 'real' and 'fake' subdirectories based on metadata labels
"""
import os
import json
import shutil
from pathlib import Path

def organize_videos():
    """Organize training videos into real and fake subdirectories"""
    
    # Define paths
    base_dir = Path(__file__).parent
    train_dir = base_dir / "dataset" / "train_sample_videos"
    metadata_file = train_dir / "metadata.json"
    
    # Create subdirectories
    real_dir = train_dir / "real"
    fake_dir = train_dir / "fake"
    
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)
    
    print("[INFO] Starting video organization...")
    print(f"[INFO] Source directory: {train_dir}")
    
    # Load metadata
    print("[INFO] Loading metadata.json...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Statistics
    real_count = 0
    fake_count = 0
    moved_count = 0
    error_count = 0
    
    # Process each video
    total_videos = len(metadata)
    print(f"[INFO] Found {total_videos} videos in metadata")
    print("[INFO] Organizing videos...")
    
    for idx, (video_name, info) in enumerate(metadata.items(), 1):
        label = info.get('label', 'UNKNOWN')
        source_path = train_dir / video_name
        
        # Skip if video file doesn't exist
        if not source_path.exists():
            print(f"[WARNING] Video not found: {video_name}")
            error_count += 1
            continue
        
        # Determine destination
        if label == "REAL":
            dest_path = real_dir / video_name
            real_count += 1
        elif label == "FAKE":
            dest_path = fake_dir / video_name
            fake_count += 1
        else:
            print(f"[WARNING] Unknown label '{label}' for {video_name}")
            error_count += 1
            continue
        
        # Move the video
        try:
            shutil.move(str(source_path), str(dest_path))
            moved_count += 1
            
            # Progress indicator
            if idx % 50 == 0:
                print(f"[PROGRESS] Processed {idx}/{total_videos} videos...")
        
        except Exception as e:
            print(f"[ERROR] Failed to move {video_name}: {e}")
            error_count += 1
    
    # Move metadata.json to root level (keep original)
    print("\n[INFO] Organization complete!")
    print("=" * 60)
    print(f"[SUMMARY] Total videos processed: {total_videos}")
    print(f"[SUMMARY] REAL videos: {real_count} (moved to 'real' folder)")
    print(f"[SUMMARY] FAKE videos: {fake_count} (moved to 'fake' folder)")
    print(f"[SUMMARY] Successfully moved: {moved_count}")
    print(f"[SUMMARY] Errors/Warnings: {error_count}")
    print("=" * 60)
    print(f"\n[SUCCESS] Videos organized into:")
    print(f"  - {real_dir}")
    print(f"  - {fake_dir}")
    print(f"\n[NOTE] metadata.json remains in: {train_dir}")

if __name__ == "__main__":
    try:
        organize_videos()
    except Exception as e:
        print(f"[ERROR] Script failed: {e}")
        import traceback
        traceback.print_exc()
