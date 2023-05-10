import io
import os
import sys
import random
from typing import Union
from random import randint

import numpy as np
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from tqdm import tqdm

from contour import Contourizer

random.seed()


class PathAgent:
    def __init__(self, ymax: int, xmax: int, x0: int = None, x1: int = None,
                 qx: int = None, y0: int = None, y1: int = None, qy: int = None):
        if x0 is not None:
            if x1 is not None and x0 + x1 > xmax:
                x1 = xmax - x0
            if qx is not None and x0 + qx > xmax:
                qx = xmax - x0
        if y0 is not None:
            if y1 is not None and y0 + y1 > ymax:
                y1 = ymax - y0
            if qy is not None and y0 + qy > ymax:
                qy = ymax - y0
        self.x0 = randint(0, xmax) if x0 is None else x0
        self.x1 = randint(-self.x0, xmax - self.x0) if x1 is None else x1
        self.y0 = randint(0, ymax) if y0 is None else y0
        self.y1 = randint(-self.y0, ymax - self.y0) if y1 is None else y1
        self.qx = randint(-self.x0, xmax - self.x0) if qx is None else qx
        self.qy = randint(-self.y0, ymax - self.y0) if qy is None else qy
        self.xmax = xmax
        self.ymax = ymax
        self.chromosome = {'x0': self.x0, 'x1': self.x1, 'qx': self.qx,
                           'y0': self.y0, 'y1': self.y1, 'qy': self.qy}
        self.path = '''<path d="M {x0} {y0} q {qx} {qy} {x1} {y1}" 
                            stroke="{c}" stroke_width="1" fill="none"/>'''
        self.fitness = sys.maxsize

    def crossover(self, other_agent, mutation_rate=0.1):
        child_chromosome = {}
        inheritance_rate = (1 - mutation_rate) / 2
        p = [inheritance_rate, inheritance_rate, mutation_rate]
        gene_pool = [self.chromosome, other_agent.chromosome]
        gene_keys = self.chromosome.keys()
        choices = np.random.choice(3, len(gene_keys), p=p)
        for k, chosen in zip(gene_keys, choices):
            if chosen != 2:
                child_chromosome[k] = gene_pool[chosen][k]
        return PathAgent(ymax=self.ymax, xmax=self.xmax, **child_chromosome)

    def get_path(self, c='black'):
        return self.path.format(x0=self.x0, x1=self.x1,
                                y0=self.y0, y1=self.y1,
                                qx=self.qx, qy=self.qy,
                                c=c)

    def __str__(self):
        return self.get_path()


class SVG:
    def __init__(self,
                 contourizer: Contourizer = Contourizer(),
                 mutation_rate=0.1,
                 n_buffer=5):
        """
        Векторизатор изображений

        Параметры:
            contourizer (Contourizer, optional): Объект векторизатора. По умолчанию Contourizer().
        """
        self.contourizer = contourizer
        self.mutation_rate = mutation_rate
        self.code = '<svg width="{w}" height="{h}">{x}</svg>'
        self.n_buffer = n_buffer

    def assess_fitness(self, agent):
        svg_code = self.get_svg([agent], colors=['black'])
        buffer = self.svg_to_pngbuffer(svg_code)
        agent_img = np.array(Image.open(buffer))
        agent.fitness = (agent_img != self.img).mean()

    def svg_to_pngbuffer(self, svg_code):
        buffer = io.StringIO()
        buffer.write(svg_code)
        buffer.seek(0)
        svg = svg2rlg(buffer)
        buffer = io.BytesIO()
        renderPM.drawToFile(svg, buffer, fmt='PNG')
        return buffer

    def visualize(self, i, n):
        colors = ['red'] + ['blue'] * (n-1)
        svg_code = self.get_svg(self.population[:n], colors)
        with open(os.path.join(self.resdir, f'{i}.png'), 'wb') as f:
            f.write(self.svg_to_pngbuffer(svg_code).getbuffer())

    def __call__(self, img: Union[str, np.ndarray],
                 n_agents=10,
                 seed=None,
                 gain=0.1,
                 max_epochs=1000,
                 resroot=None) -> None:
        """
        Преобразовать картинку в SVG-код

        Параметры:
            img (Union[str, np.ndarray]): картинка на вход (путь или матрица[HWC])
            n_agents (int): Кол-во агентов. По умолчанию 10
            seed (int): Ядро рандомизатора чисел. По умолчанию случайное
            n_epochs (int): Кол-во эпох. По умолчанию 1000
        """
        if resroot:
            self.resdir = os.path.join(
                resroot, f'test{len(os.listdir(resroot))}')
            os.mkdir(self.resdir)
        else:
            self.resdir = None
        random.seed(seed)
        self.img = np.array(self.contourizer.contour(img, invert=True))
        self.h, self.w, _ = self.img.shape
        self.population = [PathAgent(xmax=self.h, ymax=self.w)
                           for _ in range(n_agents)]
        self.n_pixels = np.prod(self.img.shape)
        n_agents_halfed = n_agents // 2
        fitness_buffer = []
        for i in tqdm(range(max_epochs)):
            for agent in self.population:
                self.assess_fitness(agent)
            if self.resdir:
                self.visualize(i, n=5)
            self.population.sort(key=lambda x: x.fitness)
            if self.population[0].fitness <= 0:
                break
            n_best = int(n_agents * gain)
            new_generation = self.population[:n_best]
            n_children = n_agents - n_best
            fitness_buffer.insert(0, self.population[0].fitness)
            if len(fitness_buffer) > self.n_buffer:
                fitness_buffer.pop()
                if len(set(fitness_buffer)) == 1:
                    break
            for _ in range(n_children):
                parent1 = random.choice(self.population[:n_agents_halfed])
                parent2 = random.choice(self.population[:n_agents_halfed])
                new_generation.append(
                    parent1.crossover(parent2,
                                      mutation_rate=self.mutation_rate))
            self.population = new_generation
        print("Generation: {}\tChromosome: {}\tFitness: {}".
              format(i, self.population[0].chromosome,
                     self.population[0].fitness))

    def get_svg(self, agents, colors):
        return self.code.format(w=self.w, h=self.h,
                                x=''.join([a.get_path(c=c)
                                           for c, a in zip(colors, agents)]))

    def __str__(self):
        return self.get_svg(str(self.population[0], colors=['black']))

    def export_svg(self, outsvg: str) -> None:
        """Экспортировать как svg-файл

        Параметры:
            outsvg (str): путь к файлу результата
        """
        with open(outsvg, 'w+') as f:
            f.write(self.__str__())
