"""
Pygame implementation of reaction-diffusion.

'esc' to hide/show the sliders
click to add more substance
'r' to reset
"""
from itertools import starmap

import cv2
import numpy as np

import pygame
import pygame.freetype
from pygame.freetype import Font


WINDOW_DIM = [500, 500]

DROP = np.array([[0., 0., 1., 1., 1., 1., 1., 0., 0.],
                 [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                 [0., 0., 1., 1., 1., 1., 1., 0., 0.],])


class Slider():
    """
    Slider code lifted from
    https://www.dreamincode.net/forums/topic/401541-buttons-and-sliders-in-pygame/
    """

    def __init__(self, name, val, min_max, pos, window, dim=[150, 30]):
        self.name = name
        self.val = val
        self.min_v, self.max_v = min_max
        self.xpos, self.ypos = pos
        self.window = window
        self.width, self.height = dim
        self.hit = False

        #Slider background
        self.surf = pygame.surface.Surface((self.width, self.height))
        self.surf.set_alpha(200)
        self.surf.fill((100, 100, 100))

        pygame.draw.rect(self.surf, (255, 255, 255), [1, 1, self.width - 2, self.height - 2], 1)
        pygame.draw.rect(self.surf, (0, 0, 0), [5, self.height - 9, self.width - 10, 2], 0)
        pygame.draw.rect(self.surf, (255, 255, 255), [5, self.height - 10, self.width - 10, 4], 1)

        #Slider text
        self.FONT = Font('NotoSansMono-Regular.ttf', 10)
        self.txt_surf, self.txt_rect = self.FONT.render(self.name + f'{self.val:.4}', (255, 255, 255))

        #Slider button
        self.button_surf = pygame.surface.Surface((20, 20))
        self.button_rect = self.button_surf.get_rect()
        self.button_surf.fill((1, 1, 1))
        self.button_surf.set_colorkey((1, 1, 1))
        pygame.draw.circle(self.button_surf, (255, 255, 255), (10, 10), 3)
        pygame.draw.circle(self.button_surf, (0, 0, 0), (10, 10), 2)

    def draw(self):
        """
        Draws the slider.
        """
        surf = self.surf.copy()
        x = 5 + int((self.val - self.min_v) / (self.max_v - self.min_v) * (self.width - 10))
        y = self.height - 8
        self.button_rect = self.button_surf.get_rect(center=(x, y))
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)

        x, y = self.surf.get_rect().center
        self.txt_rect = self.txt_surf.get_rect(center=(x, y - 4))

        surf.blit(self.txt_surf, self.txt_rect)
        self.window.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """
        Moves the slider handle and updates text.
        """
        self.txt_surf, _ = self.FONT.render(self.name + f'{self.val:.4}', (255, 255, 255))
        self.val = ((pygame.mouse.get_pos()[0] - self.xpos - 5) /
                    (self.width - 10) * (self.max_v - self.min_v)) + self.min_v
        self.val = np.clip(self.val, self.min_v, self.max_v)


class ReactDiffuse:
    diffusion_of_A=1.
    diffusion_of_B=.5
    feed=.01624
    kill=.04465

    laplace_A = np.zeros(WINDOW_DIM, dtype=np.float32)
    laplace_B = np.zeros(WINDOW_DIM, dtype=np.float32)
    react_chance = np.zeros(WINDOW_DIM, dtype=np.float32)
    new_A = np.zeros(WINDOW_DIM, dtype=np.float32)
    new_B = np.zeros(WINDOW_DIM, dtype=np.float32)
    difference = np.zeros(WINDOW_DIM, dtype=np.float32)
    kernel = np.array([[.05, .2, .05],
                       [0.2, -1, 0.2],
                       [.05, .2, .05]])

    hide_sliders = False
    mouse_down = False
    running = True

    def __init__(self):
        self.window = window = pygame.display.set_mode(WINDOW_DIM)

        sliders = (("feed = ", self.feed, [.001, .08], [20, 20], window),
                   ("kill = ", self.kill, [.01, .073], [20, 52], window),
                   ("diffusion of A = ", self.diffusion_of_A, [.8, 1.2], [20, 84], window),
                   ("diffusion of B = ", self.diffusion_of_B, [.4, .6], [20, 116], window))
        self.sliders = list(starmap(Slider, sliders))

    def step(self):
        """
        Vectorized implementation of the Gray-Scott algorithm.

        Read more here:
        https://www.algosome.com/articles/reaction-diffusion-gray-scott.html
        """
        A = self.A
        B = self.B

        cv2.filter2D(A, ddepth=-1, kernel=self.diffusion_of_A * self.kernel,
                     dst=self.laplace_A, borderType=2)
        cv2.filter2D(B, ddepth=-1, kernel=self.diffusion_of_B * self.kernel,
                     dst=self.laplace_B, borderType=2)

        #React chance is used in both equations
        np.multiply(B, B, out=self.react_chance)
        np.multiply(A, self.react_chance, out=self.react_chance)
        #First Equation done in place
        np.multiply(A, 1 - self.feed, out=self.new_A)
        np.add(self.new_A, self.laplace_A, out=self.new_A)
        np.subtract(self.new_A, self.react_chance, out=self.new_A)
        np.add(self.new_A, self.feed, out=self.new_A)
        #Second Equation done in place
        np.multiply(self.B, 1 - self.kill - self.feed, out=self.new_B)
        np.add(self.new_B, self.laplace_B, out=self.new_B)
        np.add(self.new_B, self.react_chance, out=self.new_B)
        #Clip arrays
        np.clip(self.new_A, 0, 1, out=A)
        np.clip(self.new_B, 0, 1, out=B)

    def color(self):
        """
        Currently we are scaling the difference, B-A, to be between 0 and 255,
        then we stack that value 3 times to create a (R,G,B) color.
        """
        np.subtract(self.B, self.A, out=self.difference)
        np.add(self.difference, 1, out=self.difference)
        np.multiply(self.difference, 127.5, out=self.difference)
        return np.dstack([self.difference]*3)

    def get_user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_ESCAPE:
                    self.hide_sliders = not self.hide_sliders
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
                if event.button == pygame.BUTTON_LEFT:
                    pos = pygame.mouse.get_pos()
                    for slider in self.sliders:
                        if slider.button_rect.collidepoint(pos):
                            slider.hit = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False
                for slider in self.sliders:
                    slider.hit = False

                self.feed = self.sliders[0].val
                self.kill = self.sliders[1].val
                self.diffusion_of_A = self.sliders[2].val
                self.diffusion_of_B = self.sliders[3].val

    def add_substance(self):
        if not any(slider.hit for slider in self.sliders):
            try:
                x, y = pygame.mouse.get_pos()
                self.B[x - 4:x + 5, y - 4:y + 5] = DROP
            except ValueError:
                #Too close to border
                pass

    def reset(self):
        self.A = np.ones(WINDOW_DIM, dtype=np.float32)
        self.B = np.zeros(WINDOW_DIM, dtype=np.float32)
        x, y = WINDOW_DIM
        self.B[x // 2 - 10: x // 2 + 11, y // 2 - 10: y // 2 + 11] = 1

    def draw_sliders(self):
        for slider in self.sliders:
            if slider.hit:
                slider.move()
            slider.draw()

    def start(self):
        self.reset()

        while self.running:
            self.step()
            pygame.surfarray.blit_array(self.window, self.color())
            if not self.hide_sliders:
                self.draw_sliders()
            if self.mouse_down:
                self.add_substance()
            pygame.display.update()
            self.get_user_input()


def main():
    pygame.init()
    pygame.display.set_caption('reaction-diffusion')
    ReactDiffuse().start()
    pygame.quit()

if __name__ == "__main__":
    main()
