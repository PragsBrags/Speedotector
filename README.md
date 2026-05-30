# Speedotector

Speedotector is a compact Python computer-vision project that detects license plates in video, crops plate regions, and reads the plate text using OCR.

The demo uses `test2.mp4` (project root), a YOLO license-plate model at `models/license_plate.pt`, OpenCV for video processing, and PaddleOCR for text recognition.

## Project Pipeline

1. `main.py` opens `test2.mp4` and saves video metadata to the local database.
2. `ingestion/video_feed.py` scans the video and selects useful frames with motion and sharpness checks.
3. `ocr/licensePlate.py` loads the YOLO model and finds the best license-plate bounding box in each selected frame.
4. The detected plate region is cropped and lightly preprocessed for OCR.
5. PaddleOCR reads the cropped plate image.
6. The detected plate text and confidence values are printed in the terminal and persisted to the local DB (`db/`).

Note: To change the demo video or model, update the `video_path` and `model_path` variables in `main.py` (see Configuration section).

## Project Structure

```text
.
|-- docker/
|   |-- Dockerfile
|   `-- docker-compose.yml
|-- db/
|   |-- __init__.py
|   |-- database.py
|   `-- models.py
|-- ingestion/
|   |-- __init__.py
|   |-- cropping.py
|   `-- video_feed.py
|-- models/
|   `-- license_plate.pt
|-- ocr/
|   |-- __init__.py
|   |-- licensePlate.py
|   `-- vehicleDetection.py
|-- main.py
|-- requirements.txt
`-- test2.mp4
```


## Run With Docker

Make sure Docker Desktop (or the Docker Engine) is running.

Windows (PowerShell):

```powershell
docker compose -f docker\docker-compose.yml up --build app
```

To stop:

```powershell
docker compose -f docker\docker-compose.yml down
```

Run Jupyter Lab (optional):

```powershell
docker compose -f docker\docker-compose.yml --profile lab up --build lab
```

Linux/macOS (bash):

```bash
docker compose -f docker/docker-compose.yml up --build app
```

Stop:

```bash
docker compose -f docker/docker-compose.yml down
```

Run Jupyter Lab (optional):

```bash
docker compose -f docker/docker-compose.yml --profile lab up --build lab
```

Notes:
- The repo uses modern `docker compose` syntax (Compose v2). If you have an older `docker-compose` binary, adapt the command.
- The first build may take a long time due to large ML dependencies (PyTorch/PaddleOCR/OpenCV).

## Run Locally Without Docker

### System requirements

- Python 3.8+ recommended.
- If you plan to use GPU acceleration, install appropriate CUDA/cuDNN versions for your `paddlepaddle`/PyTorch setup.

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

### Linux/macOS

```bash
python3 -m venv venv_paddle
source venv_paddle/bin/activate
pip install -r requirements.txt
python main.py
```

## Configuration

- Default video path: edit `video_path` in `main.py` (defaults to `test2.mp4`).
- Default model path: edit `model_path` in `main.py` (defaults to `models/license_plate.pt`).
- To add CLI flags or environment-based config, consider a small wrapper (I can add an `argparse` interface if you want).

## Troubleshooting

- "Could not open video": ensure the video file exists and the path in `main.py` is correct.
- Missing model errors: place the YOLO model at `models/license_plate.pt` or update `model_path`.
- CPU vs GPU: Paddle and detection models may require CPU-only installs or GPU-enabled packages — check `paddlepaddle` installation docs for your platform.
- Docker build is slow: try a cached build or pre-install large dependencies on your host.

## Example output

```
Video opened successfully.
Video saved to database with id: 1
License Plate Detection model loaded successfully.
Paddle OCR model loaded successfully.
Plate crop size: 240x64px
OCR result:
 [('ABC1234', 0.92)]
Detection saved to database with id: 5
```

## Contributing

If you'd like contributions, add a `CONTRIBUTING.md` and a license. Typical next steps:
- Add a short test that checks `main.py` runs with a very short sample video.
- Add CI (GitHub Actions) to run linters/tests on PRs.

## Responsible use

This project processes vehicle/license-plate imagery. Ensure you comply with local privacy laws and obtain permission before processing identifying data.
