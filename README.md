# Speedotector

Speedotector is a Python computer-vision project for detecting license plates in a video and reading the plate text with OCR.

The current demo uses `test_video.mp4`, a YOLO license-plate model at `models/license_plate.pt`, OpenCV for video/frame processing, and PaddleOCR for text recognition.

This project can be run in two ways:

- `main.py` runs the original command-line pipeline.
- `app.py` runs the Streamlit web UI for uploading a video, selecting a region of interest, and viewing detections.

## Project Pipeline

1. `main.py` opens `test_video.mp4`.
2. `ingestion/video_feed.py` scans the video and selects useful frames with motion and sharpness checks.
3. `ocr/licensePlate.py` loads the YOLO model and finds the best license-plate bounding box in each selected frame.
4. The detected plate region is cropped and lightly preprocessed for OCR.
5. PaddleOCR reads the cropped plate image.
6. The detected plate text and confidence values are printed in the terminal.

## Project Structure

```text
.
|-- docker/
|   |-- Dockerfile
|   `-- docker-compose.yml
|-- ingestion/
|   `-- video_feed.py
|-- models/
|   `-- license_plate.pt
|-- ocr/
|   |-- licensePlate.py
|   `-- vehicleDetection.py
|-- app.py
|-- main.py
|-- pipeline.py
|-- requirements.txt
`-- test_video.mp4
```

## Run With Docker

Make sure Docker Desktop is running first.

### Windows

From the project root, build and run the app:

```powershell
docker compose -f docker\docker-compose.yml up --build app
```

To stop it:

```powershell
docker compose -f docker\docker-compose.yml down
```

To run Jupyter Lab (optional):

```powershell
docker compose -f docker\docker-compose.yml --profile lab up --build lab
```

### Linux/macOS

From the project root, build and run the app:

```bash
docker compose -f docker/docker-compose.yml up --build app
```

To stop it:

```bash
docker compose -f docker/docker-compose.yml down
```

To run Jupyter Lab (optional):

```bash
docker compose -f docker/docker-compose.yml --profile lab up --build lab
```

Then open the Jupyter URL printed in the terminal.

## Run Locally Without Docker

### Windows

Create and activate a virtual environment:

```powershell
python -m venv venv_paddle
.\venv_paddle\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the pipeline:

```powershell
python main.py
```

Run the Streamlit web UI:

```powershell
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Linux/macOS

Create and activate a virtual environment:

```bash
python3 -m venv venv_paddle
source venv_paddle/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python main.py
```

Run the Streamlit web UI:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Using the Streamlit Web UI

1. Upload a video file (`mp4`, `mov`, `avi`, or `mkv`).
2. Keep **Use full frame** checked to process the whole frame, or uncheck it to select a region of interest.
3. When selecting a region of interest, draw a rectangle on the first-frame canvas or enter exact `x`, `y`, `width`, and `height` values manually.
4. Choose whether to save detections to the database.
5. Click **Run detection**.

The app shows one first-frame ROI preview by default so the page stays within the visible screen height. Expand **Playback** if you want to inspect the full video.

## Notes

- The first Docker build can take a long time because PyTorch, PaddleOCR, OpenCV, and Jupyter are large dependencies.
- `main.py` currently reads `test_video.mp4` from the project root.
- `app.py` requires uploading a video through the browser; it does not automatically load `test_video.mp4`.
- The YOLO model file must exist at `models/license_plate.pt`.
- OCR debug images are disabled by default. Enable `PaddleInference(debug=True)` if you need them during development.
