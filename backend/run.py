import os
import time
import chardet

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from vectorizer.genetic import SVG
from contour import Contourizer


app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"], # Allows all methods
#     allow_headers=["*"], # Allows all headers
# )

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.post("/upload")
async def upload(image: UploadFile = File(...), max_epochs : str='100'):
    # Save the received image to a file
    max_epochs = int(max_epochs)
    with open("received_image.jpg", "wb") as file:
        file.write(image.file.read())
    contour = Contourizer()
    svg = SVG(contour, n_buffer=100, mutation_rate=0.2, resroot='results', n_agents=10, max_epochs=max_epochs)
    svg("received_image.jpg")
    svg_path = os.path.join(svg.resdir, 'result.svg')
    with open(svg_path, 'r+') as f:
        svg_content = f.read()
    # Return a different image as the response
    return {'image': svg_content}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)