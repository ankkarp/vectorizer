import io
import os
import sys
import random
import shutil
from typing import Union
from random import randint

import numpy as np
from PIL import Image, ImageDraw
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from tqdm import tqdm

from contour import Contourizer

random.seed()


class PathAgent:
    def __init__(self, ymax: int, xmax: int, x0: int = None, xc0: int = None,
                 x1: int = None, xc1: int = None, y0: int = None,
                 yc0: int = None, y1: int = None, yc1: int = None):
        self.xmax = xmax
        self.ymax = ymax
        self.x0 = randint(0, xmax) if x0 is None else x0
        self.xc0 = randint(0, xmax) if xc0 is None else xc0
        self.x1 = randint(0, xmax) if x1 is None else x1
        self.xc1 = randint(0, xmax) if xc1 is None else xc1
        self.y0 = randint(0, ymax) if y0 is None else y0
        self.yc0 = randint(0, ymax) if yc0 is None else yc0
        self.y1 = randint(0, ymax) if y1 is None else y1
        self.yc1 = randint(0, ymax) if yc1 is None else yc1
        self.chromosome = {'x0': self.x0, 'xc0': self.xc0, 'x1': self.x1, 'xc1': self.xc1,
                           'y0': self.y0, 'yc0': self.yc0, 'y1': self.y1, 'yc1': self.yc1, }
        self.path = '''<path d="M {x0} {y0} C {xc0} {yc0}, {xc1} {y1}, {x1} {y1}"
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
        return self.path.format(x0=self.x0, x1=self.x1, xc0=self.xc0, xc1=self.xc1,
                                y0=self.y0, y1=self.y1, yc0=self.yc0, yc1=self.yc1,
                                c=c)

    def __str__(self):
        return self.get_path()


class SVG:
    def __init__(self,
                 contourizer: Contourizer = Contourizer(),
                 mutation_rate=0.1,
                 n_buffer=100):
        """
        Векторизатор изображений

        Параметры:
            contourizer (Contourizer, optional): Объект векторизатора. По умолчанию Contourizer().
        """
        self.contourizer = contourizer
        self.mutation_rate = mutation_rate
        self.code = '<svg width="{w}" height="{h}">{x}</svg>'
        self.n_buffer = n_buffer

    def assess_fitness(self, agent, method='dice'):
        svg_code = self.get_svg([agent], colors=['black'])
        buffer = self.svg_to_pngbuffer(svg_code)
        agent_img = np.array(Image.open(buffer).convert("L"))
        if method == 'curve':
            pred_curve_idxs = np.where(agent_img.flatten() < 127)[0]
            match_idxs = np.intersect1d(pred_curve_idxs, self.true_curve_idxs)
            match_n = len(match_idxs)
            fn = self.true_curve_n - match_n
            fp = len(pred_curve_idxs) - len(match_idxs)
            agent.fitness = fn
        elif method == 'naive':
            agent.fitness = (agent_img != self.img).mean()
        elif method == 'dice':
            pred_curve_idxs = np.where(agent_img.flatten() < 127)[0]
            match_idxs = np.intersect1d(pred_curve_idxs, self.true_curve_idxs)
            tp = len(match_idxs)
            n_all = len(pred_curve_idxs) + len(self.true_curve_idxs)
            agent.fitness = 1 - 2 * tp / n_all

    def svg_to_pngbuffer(self, svg_code):
        buffer = io.StringIO()
        buffer.write(svg_code)
        buffer.seek(0)
        svg = svg2rlg(buffer)
        buffer = io.BytesIO()
        renderPM.drawToFile(svg, buffer, fmt='PNG')
        return buffer

    def visualize(self, i, n):
        colors = ['blue'] * (n - 1) + ['red']
        best_specimen = self.population[:n][::-1]
        svg_code = self.get_svg(best_specimen, colors)
        img_buffer = self.svg_to_pngbuffer(svg_code)
        img = Image.open(img_buffer)
        drawing = ImageDraw.Draw(img)
        drawing.text((2, 2), self.text_template % (i, best_specimen[0].fitness),
                     fill=(255, 255, 255))
        img.save(self.img_template % i)
        # with open(img_path, 'wb') as f:
        #     f.write(self.svg_to_pngbuffer(svg_code).getbuffer())

    def makegif(self):
        palette_path = os.path.join(self.resdir, 'palette.png')
        os.system('ffmpeg -i {} -vf palettegen {}'.format(
            self.img_template,
            palette_path,
        ))
        os.system("ffmpeg -i {} -i {} -lavfi paletteuse -framerate 5 -loop 0 {}".format(
            self.img_template,
            palette_path,
            os.path.join(self.resdir, 'output.gif')
        ))
        shutil.rmtree(f'{self.tempdir}')

    def __call__(self, img: Union[str, np.ndarray],
                 n_agents=10,
                 seed=None,
                 gain=0.1,
                 max_epochs=1000,
                 resroot=None,
                 outfile='result.svg',
                 fitness_method='dice',
                 preserve_dublicates=False) -> None:
        """
        Преобразовать картинку в SVG-код

        Параметры:
            img (Union[str, np.ndarray]): картинка на вход (путь или матрица[HWC])
            n_agents (int): Кол-во агентов. По умолчанию 10
            seed (int): Ядро рандомизатора чисел. По умолчанию случайное
            n_epochs (int): Кол-во эпох. По умолчанию 1000
        """
        i = 0
        if resroot:
            while True:
                resdir = f'{fitness_method}{i}'
                if resdir not in os.listdir(resroot):
                    break
                i += 1
            self.resdir = os.path.join(resroot, resdir)
            self.tempdir = os.path.join(self.resdir, 'temp')
            n_digits = int(np.ceil(np.log10(max_epochs + 1)))
            self.img_template = f'{self.tempdir}/%0{n_digits}d.png'
            self.text_template = f'%0{n_digits}d: %d.png'
            os.makedirs(self.tempdir, exist_ok=False)
        random.seed(seed)
        pil_img = self.contourizer.contour(img, invert=True)
        pil_img.save(os.path.join(self.resdir, f'contour.png'))
        self.img = np.array(pil_img.convert("L"))
        self.flat_img = self.img.flatten()
        self.true_curve_idxs = np.where(self.flat_img < 127)[0]
        self.true_curve_n = len(self.true_curve_idxs)
        self.h, self.w = self.img.shape
        self.population = [PathAgent(xmax=self.h, ymax=self.w)
                           for _ in range(n_agents)]
        self.n_pixels = np.prod(self.img.shape)
        n_agents_halfed = n_agents // 2
        fitness_buffer = []
        n_best = int(n_agents * gain)
        for i in tqdm(range(max_epochs)):
            for agent in self.population:
                self.assess_fitness(agent, fitness_method)
            if self.resdir:
                self.visualize(i, n=n_best)
            self.population.sort(key=lambda x: x.fitness)
            if self.population[0].fitness <= 0:
                break
            if preserve_dublicates:
                new_generation = self.population[:n_best]
            else:
                j = 0
                new_generation = []
                new_generation_paths = set()
                while j != n_best:
                    current_agent_path = str(self.population[j])
                    if current_agent_path not in new_generation_paths:
                        new_generation_paths.add(current_agent_path)
                        new_generation.append(self.population[j])
                    j += 1
            n_children = int(n_agents - n_best)
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
        if resroot:
            self.makegif()
            self.export_svg(os.path.join(self.resdir, outfile))
        else:
            self.export_svg(outfile)

    def get_svg(self, agents, colors):
        return self.code.format(w=self.w, h=self.h,
                                x=''.join([a.get_path(c=c)
                                           for c, a in zip(colors, agents)]))

    def __str__(self):
        return self.get_svg([self.population[0]], colors=['black'])

    def export_svg(self, outsvg: str) -> None:
        """Экспортировать как svg-файл

        Параметры:
            outsvg (str): путь к файлу результата
        """
        with open(outsvg, 'w+') as f:
            f.write(self.__str__())
