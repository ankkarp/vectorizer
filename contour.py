from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance


class Contourizer:
    def __init__(self, sigma=0.5, threshold=0.3):
        self.sigma = sigma
        self.x, self.y = self.generate_mask(threshold, sigma)
        self.gauss = self.gaussian(self.x, self.y, sigma)
        self.grad = self.calculate_gradient()
        self.gx = self.gradint(self.x)
        self.gy = self.gradint(self.y)

    def generate_mask(self, threshold, sigma):
        mask_halfsize = np.round(
            np.sqrt(-np.log(threshold) * 2 * (sigma ** 2)))
        x, y = np.meshgrid(range(-int(mask_halfsize), int(mask_halfsize) + 1),
                           range(-int(mask_halfsize), int(mask_halfsize) + 1))
        return x, y

    def gaussian(self, x, y, sigma):
        temp = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
        return (np.exp(-temp))

    def calculate_gradient(self):
        der = (self.x ** 2 + self.y ** 2) / (2 * self.sigma ** 2)
        return -(np.exp(-der) / self.sigma ** 2)

    def gradint(self, x):
        return np.around(-x * self.grad * 255)

    def pad(self, img, kernel):
        r, c = img.shape
        kr, kc = kernel.shape
        padded = np.zeros((r + kr, c + kc), dtype=img.dtype)
        insert = np.uint((kr)/2)
        padded[insert: insert + r, insert: insert + c] = img
        return padded

    def smooth(self, img, kernel=None):
        if kernel is None:
            mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        else:
            mask = kernel
        i, j = mask.shape
        output = np.zeros((img.shape[0], img.shape[1]))
        image_padded = self.pad(img, mask)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                output[x, y] = (mask * image_padded[x:x+i,
                                                    y:y+j]).sum() / mask.sum()
        return output

    def apply_mask(self, image, kernel):
        i, j = kernel.shape
        kernel = np.flipud(np.fliplr(kernel))
        output = np.zeros_like(image)
        image_padded = self.pad(image, kernel)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                output[x, y] = (kernel * image_padded[x:x+i, y:y+j]).sum()
        return output

    def gradient_magnitude(self, fx, fy):
        mag = np.zeros((fx.shape[0], fx.shape[1]))
        mag = np.sqrt((fx ** 2) + (fy ** 2))
        mag = mag * 100 / mag.max()
        return np.around(mag)

    def contour(self, img: Union[str, np.ndarray],
                invert=True,
                outpath=None,
                enhance_rate=10,
                show=False) -> np.ndarray:
        """
        Выделить контур изображения

        Args:
            img (str | PIL.Image): картинка для обработки (путь к файлу / объект PIL.Image.Image)
        """
        plt.figure(figsize=(8, 8))
        if type(img) == str:
            img = Image.open(img)
        img = np.array(img.convert("L"))
        img = self.smooth(img, self.gauss)
        fx = self.apply_mask(img, self.gx)
        fy = self.apply_mask(img, self.gy)
        mag = Image.fromarray(
            self.gradient_magnitude(fx, fy).astype(int)).convert('RGB')
        mag = ImageEnhance.Contrast(mag).enhance(enhance_rate)
        if invert:
            mag = ImageOps.invert(mag)
        if show:
            mag.show()
        if outpath:
            mag.save(outpath)
        return mag
