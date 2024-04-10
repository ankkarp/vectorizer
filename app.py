import io
import os

import matplotlib
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

from vectorizer.genetic import SVG
from contour import Contourizer

contour = Contourizer()
svg = SVG(contour, n_buffer=100, mutation_rate=0.2, resroot='results', n_agents=100, max_epochs=100)


def convert(image):
    global svg
    svg_code = svg(image)
    buffer = io.StringIO()
    buffer.write(svg_code)
    buffer.seek(0)
    svg = svg2rlg(buffer)
    buffer = io.BytesIO()
    renderPM.drawToFile(svg, buffer, fmt='PNG')
    return np.array(Image.open(buffer))


inputs = [
    gr.Image(type="filepath"),
]

outputs = [
    gr.Image(type='numpy'),
]

demo = gr.Interface(convert, inputs, outputs)
matplotlib.use('TkAgg')

demo.launch(share=True)