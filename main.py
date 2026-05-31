import os

from pipeline import process_video


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.getenv("VIDEO_PATH", os.path.join(script_dir, "test_video.mp4"))

    results = process_video(video_path=video_path, save_to_db=True)
    print(f"Finished processing {video_path}. Found {len(results)} detection(s).")

    for result in results:
        detection_id = result.get("detection_id")
        print(
            f"Detection {detection_id or result['frame_index']}: "
            f"{result['plate_text']} at {result['coords']}"
        )


if __name__ == "__main__":
    main()
