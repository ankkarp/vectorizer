import os
from pathlib import Path

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from backend.vectorizer.genetic import SVG
from backend.vectorizer.contour import Contourizer

app = FastAPI()


@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.post("/upload")
async def upload(image: UploadFile, max_epochs: int):
    with open("received_image.jpg", "wb") as file:
        file.write(image.file.read())
    contour = Contourizer()
    svg = SVG(contour, n_buffer=100, mutation_rate=0.2, resroot='results', n_agents=100, max_epochs=max_epochs)
    svg("received_image.jpg")
    svg_path = os.path.join(svg.resdir, 'result.svg')
    with open(svg_path, 'r+') as f:
        svg_content = f.read()
    return {'image': svg_content, 'resdir': Path(svg.resdir).name}

@app.get('/process_gif/{res_dir}')
async def process_gif(res_dir: str):
    return FileResponse(os.path.join('results', res_dir, 'output.gif'), media_type='image/gif')

@app.get('/contour/{res_dir}')
async def contour(res_dir: str):
    Image.open(os.path.join('results', res_dir, 'contour.png'))
    return FileResponse(os.path.join('results', res_dir, 'contour.png'), media_type='image/png')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)