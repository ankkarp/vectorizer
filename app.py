import io
import os

import matplotlib
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


def convert(image):
    svg_code = ...
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
    gr.Image(type='filepath'),
]

demo = gr.Interface(convert, inputs, outputs)
matplotlib.use('TkAgg')

demo.launch(share=True)