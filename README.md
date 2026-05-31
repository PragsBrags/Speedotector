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

Make sure Docker Desktop is running first.

### Windows

From the project root, build and run the application:

```powershell
docker compose -f docker\docker-compose.yml up --build
```

Open the dashboard:

```text
http://localhost:8000
```

Open the API documentation:

```text
http://localhost:8000/docs
```

To stop the application:

```powershell
docker compose -f docker\docker-compose.yml down
```

### Linux/macOS

From the project root, build and run the application:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Open the dashboard:

```text
http://localhost:8000
```

Open the API documentation:

```text
http://localhost:8000/docs
```

To stop the application:

```bash
docker compose -f docker/docker-compose.yml down
```

---

## Run Detection From Dashboard

1. Open:

```text
http://localhost:8000/docs
```

2. Find:

```text
POST /run-detection
```

3. Click:

```text
Try it out
```

4. Click:

```text
Execute
```

The system will then:

* Process the video
* Detect license plates using YOLO
* Extract text using PaddleOCR
* Save detections to MySQL
* Generate latest detection images in the dashboard

---

## Run Locally Without Docker

### Windows

Create and activate a virtual environment:

```powershell
python -m venv venv311
.\venv311\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Update `.env` for local MySQL usage:

```env
DB_HOST=localhost
DB_PORT=3307
DB_USER=root
DB_PASSWORD=12345
DB_NAME=speedotector
```

Run the application:

```powershell
python main.py
```

### Linux/macOS

Create and activate a virtual environment:

```bash
python3 -m venv venv311
source venv311/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Update `.env` for local MySQL usage:

```env
DB_HOST=localhost
DB_PORT=3307
DB_USER=root
DB_PASSWORD=12345
DB_NAME=speedotector
```

Run the application:

```bash
python main.py
```

---

## Notes

* The first Docker build may take several minutes because PyTorch, PaddleOCR, and OpenCV are large dependencies.
* PaddleOCR downloads OCR models during the first startup.
* The YOLO model file must exist at:

```text
models/license_plate.pt
```

* The input video file must exist at:

```text
test_video.mp4
```

* Detection images are automatically saved inside:

```text
outputs/
```

* The dashboard automatically refreshes to show the latest frame and detected plate crop.


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
