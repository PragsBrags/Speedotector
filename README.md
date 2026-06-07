---
title: Speedotector
sdk: docker
app_port: 7860
---

# Speedotector

Speedotector is a motion-triggered ALPR prototype. It selects useful frames from video, detects license plates using YOLO, reads plate text using PaddleOCR, and optionally saves detections.

The demo uses `test_video.mp4` (project root), a YOLO license-plate model at `models/license_plate.pt`, OpenCV for video processing, and PaddleOCR for text recognition.

- `main.py` runs the command-line ALPR pipeline.
- `app.py` runs a Streamlit web UI for uploading a video, selecting an optional region of interest, and viewing detections.

1. `main.py` opens `test_video.mp4` and saves video metadata to the local database.
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

## Hugging Face Spaces Deployment

Use a **Docker Space**. The root `Dockerfile` starts the Streamlit app on port `7860`, and the YAML block at the top of this README tells Hugging Face to build it as a Docker Space.

1. Create a Hugging Face account and a write token:

```bash
pip install -U huggingface_hub
huggingface-cli login
```

2. Create a new Space at `https://huggingface.co/new-space`.

Use these settings:

- **Space name:** `speedotector`
- **SDK:** `Docker`
- **Visibility:** `Private` while testing, or `Public` for a demo
- **Hardware:** CPU basic first; upgrade only if OCR/detection is too slow

3. Add the Space as a Git remote. Replace `<hf-username>` with your Hugging Face username.

```bash
git remote add hf https://huggingface.co/spaces/<hf-username>/speedotector
```

4. Commit and push the deployment files.

```bash
git add Dockerfile README.md .dockerignore app.py requirements.txt models/license_plate.pt yolo26n.pt
git commit -m "chore: add hugging face docker deployment"
git push hf main
```

If your current branch is not `main`, push it to the Space `main` branch:

```bash
git push hf HEAD:main
```

5. Open the Space build logs in Hugging Face. The first build can take several minutes because PaddleOCR, PaddlePaddle, OpenCV, and Ultralytics are heavy dependencies.

6. After it starts, upload a small video first and keep **Save results to database** disabled. Add database secrets only if you need persistence.

If the app is too slow on free CPU hardware, upgrade the Space hardware or move to a paid x86 VPS/GPU instance.

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

## Research Branch Additions

This branch also includes the first zone-violation research layer:

- polygon zone definitions and geometry helpers in `ingestion/zones.py`,
- vehicle detection wrappers in `detection/vehicle.py`,
- centroid tracking and plate-to-vehicle association in `tracking/`,
- traffic-signal state detection and violation rules in `rules/`,
- evidence image/result writing in `evidence/`,
- experiment metric helpers and paper-table export in `evaluation/`,
- `zones` and `violations` database tables for evidence records.
- a Streamlit zone-violation mode for drawing or entering zones, validating rule setup, running detection, and downloading evidence JSON.

Generate Markdown result tables from an experiment JSON file with:

```bash
python -m evaluation.export_tables --results outputs/results.json --format markdown
```

## Streamlit UI

1. Upload a video file (`mp4`, `mov`, `avi`, or `mkv`).
2. Choose **ALPR only** or **Zone violation detection**.

In **ALPR only** mode:

1. Keep **Use full frame** checked to process the whole frame, or uncheck it to select a region of interest.
2. When selecting a region of interest, draw a rectangle on the first-frame canvas or enter exact `x`, `y`, `width`, and `height` values manually.
3. Choose whether to save detections to the database.
4. Click **Run detection**.

In **Zone violation detection** mode:

1. Choose violation types: restricted zone, red light, or crosswalk encroachment.
2. Add zones by drawing on the first-frame canvas or by entering exact coordinates.
3. Use the setup validation messages to add required zones:
   - restricted zone requires a forbidden area,
   - red light requires a traffic light ROI and stop line,
   - crosswalk encroachment requires a crosswalk zone.
4. Download the zone configuration JSON if needed.
5. Run zone detection, then review evidence records and download the results JSON.

Uploaded videos are written to an app-managed temp directory. Replacing or clearing an upload deletes the previous temp file, and the app removes stale `speedotector_streamlit_*` temp directories older than 24 hours on startup.

## Database Notes

Detection rows store:

- plate text,
- full-frame bounding-box coordinates,
- crop size,
- detector confidence,
- OCR confidence,
- creation timestamp.

Alembic manages schema migrations:

```bash
alembic upgrade head
```

For an existing database created before Alembic was added, confirm it matches the baseline schema, then run:

Create and activate a virtual environment:

```bash
python3 -m venv venv311
source venv311/bin/activate
```

Install dependencies:

```bash
alembic stamp 20260531_0001
alembic upgrade head
```

This stamps the original tables as the baseline and applies later detection-column migrations.

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
|   |-- zones.py
|   `-- video_feed.py
|-- detection/
|   |-- plate.py
|   `-- vehicle.py
|-- tracking/
|   |-- association.py
|   `-- centroid_tracker.py
|-- rules/
|   |-- base.py
|   |-- red_light.py
|   |-- restricted_zone.py
|   |-- traffic_signal.py
|   `-- zebra_crossing.py
|-- evidence/
|   |-- overlays.py
|   `-- writer.py
|-- evaluation/
|   |-- baselines.py
|   |-- export_tables.py
|   `-- metrics.py
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
