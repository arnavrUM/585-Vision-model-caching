#!/usr/bin/env python3
"""
Helper script to create a video frames dataset JSON file.

This script helps you create a properly formatted JSON dataset for video frames.
You can use it as a template or run it to generate a sample dataset.

Usage:
    python experiment2/create_video_frames_dataset.py \
        --output video_frames_dataset.json \
        --num-frames 100 \
        --video-id video_001
"""

import argparse
import json
from pathlib import Path


def create_sample_dataset(
    output_path: str,
    num_frames: int = 100,
    video_id: str = "video_001",
    start_frame: int = 0,
    fps: float = 30.0,
    questions: list[str] | None = None,
) -> None:
    """Create a sample video frames dataset JSON file.
    
    Args:
        output_path: Path to output JSON file
        num_frames: Number of frames to generate
        video_id: Video identifier
        start_frame: Starting frame number
        fps: Frames per second (for timestamp calculation)
        questions: List of questions to cycle through (optional)
    """
    if questions is None:
        questions = [
            "What objects are visible in this frame?",
            "What is happening in this frame?",
            "Who is in this frame?",
            "What color is the main object?",
            "What is the scene type?",
        ]
    
    samples = []
    for i in range(num_frames):
        frame_num = start_frame + i
        frame_id = f"{video_id}_frame_{frame_num:04d}"
        timestamp = frame_num / fps
        
        # Cycle through questions
        question = questions[i % len(questions)]
        
        # Generate placeholder answer (you should replace with real answers)
        answer = "placeholder_answer"
        full_answer = f"This is frame {frame_num} from {video_id}."
        
        # Group frames into scenes (every 50 frames = new scene)
        scene_id = f"scene_{frame_num // 50:02d}"
        
        sample = {
            "id": frame_id,
            "imageId": frame_id,
            "question": question,
            "answer": answer,
            "fullAnswer": full_answer,
            "groups": {
                "global": "video",
                "local": scene_id,
            },
            "semantic": [
                {"operation": "query", "argument": "objects"},
                {"operation": "filter", "argument": "visible"},
            ],
            "semanticStr": "query:objects | filter:visible",
            "metadata": {
                "video_id": video_id,
                "frame_number": frame_num,
                "timestamp": round(timestamp, 2),
                "scene_id": scene_id,
            },
        }
        samples.append(sample)
    
    # Write to JSON file
    output_path_obj = Path(output_path).expanduser()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path_obj.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ Created dataset with {len(samples)} frames")
    print(f"   Output: {output_path_obj}")
    print(f"   Video ID: {video_id}")
    print(f"   Frame range: {start_frame} to {start_frame + num_frames - 1}")
    print(f"\n⚠️  Note: You need to:")
    print(f"   1. Replace 'placeholder_answer' with real answers")
    print(f"   2. Ensure image files exist matching imageId")
    print(f"   3. Set GQA_IMAGE_ROOT environment variable to image directory")


def main():
    parser = argparse.ArgumentParser(
        description="Create a video frames dataset JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a sample dataset with 100 frames
  python experiment2/create_video_frames_dataset.py \\
      --output video_frames_dataset.json \\
      --num-frames 100

  # Create dataset for specific video
  python experiment2/create_video_frames_dataset.py \\
      --output video_001_dataset.json \\
      --num-frames 500 \\
      --video-id video_001 \\
      --start-frame 0 \\
      --fps 30.0
        """,
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames to generate (default: 100)",
    )
    parser.add_argument(
        "--video-id",
        default="video_001",
        help="Video identifier (default: video_001)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame number (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for timestamp calculation (default: 30.0)",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        help="Custom questions to cycle through (optional)",
    )
    
    args = parser.parse_args()
    
    create_sample_dataset(
        output_path=args.output,
        num_frames=args.num_frames,
        video_id=args.video_id,
        start_frame=args.start_frame,
        fps=args.fps,
        questions=args.questions,
    )


if __name__ == "__main__":
    main()

