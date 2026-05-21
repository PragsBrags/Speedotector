# Speedotector

Speedotector is a Python computer-vision project for detecting license plates in a video and reading the plate text with OCR.

The current demo uses `test_video.mp4`, a YOLO license-plate model at `models/license_plate.pt`, OpenCV for video/frame processing, and PaddleOCR for text recognition.

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
|-- main.py
|-- requirements.txt
`-- test_video.mp4
```

## Run With Docker

Make sure Docker Desktop is running first.

From the project root, build and run the app:

```powershell
docker compose -f docker\docker-compose.yml up --build app
```

To stop it:

```powershell
docker compose -f docker\docker-compose.yml down
```

## Run Jupyter Lab With Docker

The Docker setup also includes an optional Jupyter Lab service:

```powershell
docker compose -f docker\docker-compose.yml --profile lab up --build lab
```

Then open the Jupyter URL printed in the terminal.

## Run Locally Without Docker

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

## Notes

- The first Docker build can take a long time because PyTorch, PaddleOCR, OpenCV, and Jupyter are large dependencies.
- `main.py` currently reads `test_video.mp4` from the project root.
- The YOLO model file must exist at `models/license_plate.pt`.
- OCR debug images may be written as `debug_plate_upscaled.jpg` and `debug_plate_processed.jpg`.
