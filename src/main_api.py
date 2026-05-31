from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from main import main
import os

app = FastAPI(title="Speedotector API")


@app.get("/")
def dashboard():
    html = """
    <html>
    <head>
        <title>Speedotector</title>
        <meta http-equiv="refresh" content="3">
    </head>
    <body style="font-family: Arial; padding: 30px;">
        <h1>🚗 Speedotector Dashboard</h1>

        <p>Status: <b>Running</b></p>

        <form action="/run-detection" method="post">
            <button style="padding: 12px 20px; font-size: 16px;">
                Run Detection
            </button>
        </form>

        <hr>

        <h2>Latest Frame</h2>
        <img src="/latest-frame?t={{timestamp}}" width="600">

        <h2>Latest Plate Crop</h2>
        <img src="/latest-plate?t={{timestamp}}" width="300">

        <p>Page refreshes every 3 seconds.</p>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/run-detection")
def run_detection():
    main()
    return {"message": "Detection completed"}


@app.get("/latest-frame")
def latest_frame():
    path = "outputs/latest_frame.jpg"
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "No frame yet"}


@app.get("/latest-plate")
def latest_plate():
    path = "outputs/latest_plate.jpg"
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "No plate yet"}