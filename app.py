import os

import matplotlib
import gradio as gr

from backend.vectorizer.genetic import SVG
from backend.vectorizer.contour import Contourizer

os.makedirs('results', exist_ok=True)

def convert(image):
    contour = Contourizer()
    svg = SVG(contour, n_buffer=100, mutation_rate=0.2, resroot='results', n_agents=100, max_epochs=100)
    svg(image)
    svg_path = os.path.join(svg.resdir, 'result.svg')
    with open(svg_path, 'r+') as f:
        svg_content = f.read()
    contour_path = os.path.join(svg.resdir, 'contour.png')
    gif_path = os.path.join(svg.resdir, 'output.gif')
    return (contour_path, gif_path), svg_content, svg_path


inputs = [
    gr.Image(type="filepath"),
]

outputs = [
    # gr.Image(label='contour'),
    # gr.Image(label='process'),
    gr.Gallery(label='contour and progress'),
    gr.Textbox(show_copy_button=True, label='svg content'),
    gr.File(label='svg download')
]

demo = gr.Interface(convert, inputs, outputs)
matplotlib.use('TkAgg')

demo.launch(share=True)