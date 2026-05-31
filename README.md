# Speedotector

Speedotector is a compact Python computer-vision project that detects license plates in video, crops plate regions, and reads the plate text using OCR.

The demo uses `test2.mp4` (project root), a YOLO license-plate model at `models/license_plate.pt`, OpenCV for video processing, and PaddleOCR for text recognition.

- `main.py` runs the command-line ALPR pipeline.
- `app.py` runs a Streamlit web UI for uploading a video, selecting an optional region of interest, and viewing detections.

1. `main.py` opens `test2.mp4` and saves video metadata to the local database.
2. `ingestion/video_feed.py` scans the video and selects useful frames with motion and sharpness checks.
3. `ocr/licensePlate.py` loads the YOLO model and finds the best license-plate bounding box in each selected frame.
4. The detected plate region is cropped and lightly preprocessed for OCR.
5. PaddleOCR reads the cropped plate image.
6. The detected plate text and confidence values are printed in the terminal and persisted to the local DB (`db/`).

Note: To change the demo video or model, update the `video_path` and `model_path` variables in `main.py` (see Configuration section).

## Requirements

- Python 3.11.
- The YOLO license-plate model at `models/license_plate.pt`.
- Optional database settings in `.env` or `DATABASE_URL` when saving detections.

If **Save results to database** is disabled in the Streamlit UI, no database connection is required.

## Run Modes

| Mode | Command | Notes |
|---|---|---|
| CLI local | `python main.py` | Uses `test_video.mp4` by default. Override with `VIDEO_PATH=/path/to/video.mp4`. |
| Web local | `streamlit run app.py` | Upload a video through the browser. Opens on `http://localhost:8501` by default. |
| CLI Docker | `docker compose -f docker/docker-compose.yml up --build app` | Runs `main.py` in the container. |
| Web Docker | `docker compose -f docker/docker-compose.yml up --build web` | Runs Streamlit on port `8501`. |
| Jupyter Docker | `docker compose -f docker/docker-compose.yml --profile lab up --build lab` | Optional notebook/lab environment on port `8888`. |

Stop Docker services with:

```bash
docker compose -f docker/docker-compose.yml down
```

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv_paddle
source venv_paddle/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv venv_paddle
.\venv_paddle\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run checks:

```bash
ruff check .
pytest
```

## Project Pipeline

1. `main.py` or `app.py` calls `process_video()` from `pipeline.py`.
2. `ingestion/video_feed.py` selects useful frames with motion and sharpness checks.
3. `ocr/licensePlate.py` uses YOLO to find the best license-plate bounding box.
4. The plate crop is preprocessed and sent to PaddleOCR.
5. Results include plate text, full-frame coordinates, detector confidence, OCR confidence, OCR segments, selected-frame metadata, and optional database IDs.

By default, pipeline results do not include raw image arrays. Streamlit requests `include_images=True` so it can display plate crops.

## Streamlit UI

1. Upload a video file (`mp4`, `mov`, `avi`, or `mkv`).
2. Keep **Use full frame** checked to process the whole frame, or uncheck it to select a region of interest.
3. When selecting a region of interest, draw a rectangle on the first-frame canvas or enter exact `x`, `y`, `width`, and `height` values manually.
4. Choose whether to save detections to the database.
5. Click **Run detection**.

Uploaded videos are written to an app-managed temp directory. Replacing or clearing an upload deletes the previous temp file, and the app removes stale `speedotector_streamlit_*` temp directories older than 24 hours on startup.

## Database Notes

Detection rows store:

- plate text,
- full-frame bounding-box coordinates,
- crop size,
- detector confidence,
- OCR confidence.

Alembic manages schema migrations:

```bash
alembic upgrade head
```

For an existing database created before Alembic was added, confirm it matches the baseline schema, then run:

```bash
alembic stamp 20260531_0001
alembic upgrade head
```

This stamps the original tables as the baseline and applies the confidence-column migration.

## Privacy Notes

Uploaded videos and detected license-plate crops can contain sensitive personal data. Keep **Save results to database** disabled unless persistence is intentional, and delete old database rows or temp files when they are no longer needed.

OCR debug images are disabled by default. If `PaddleInference(debug=True)` is enabled during development, debug files are written with unique filenames.

## Project Structure

```text
.
|-- docker/
|   |-- Dockerfile
|   `-- docker-compose.yml
|-- ingestion/
|   |-- cropping.py
|   |-- roi.py
|   `-- video_feed.py
|-- db/
|   |-- database.py
|   |-- models.py
|   `-- repository.py
|-- models/
|   `-- license_plate.pt
|-- migrations/
|   |-- env.py
|   `-- versions/
|-- ocr/
|   `-- licensePlate.py
|-- alembic.ini
|-- tests/
|-- app.py
|-- main.py
|-- pipeline.py
|-- requirements.txt
`-- test_video.mp4
```
