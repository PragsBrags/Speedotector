import os

from pipeline import process_video


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    video_path = os.getenv("VIDEO_PATH", os.path.join(script_dir, "test_video.mp4"))
    model_path = os.getenv(
        "LICENSE_PLATE_MODEL_PATH",
        os.path.join(script_dir, "models", "license_plate.pt"),
    )
    vehicle_model_path = os.getenv("VEHICLE_MODEL_PATH", "yolo26n.pt")

    results = process_video(
        video_path=video_path,
        model_path=model_path,
        vehicle_model_path=vehicle_model_path,
        save_to_db=True,
    )

    print(f"Finished processing {video_path}. Found {len(results)} detection(s).")

    for result in results:
        detection_id = result.get("detection_id")
        print(
            f"Detection {detection_id or result['frame_index']}: "
            f"{result['vehicle_class']} ({result['vehicle_confidence']:.2%}) | "
            f"{result['plate_text']} at {result['coords']}"
        )


if __name__ == "__main__":
    main()