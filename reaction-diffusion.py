#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pygame implementation of reaction-diffusion.

'esc' to hide/show the sliders
click to add more substance
'r' to reset
"""
import types
import numpy as np
import pygame
import pygame.freetype
import cv2
#import scipy.ndimage as nd   #alternative to cv2 for convolution


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
        pygame.draw.rect(self.surf, (255, 255, 255),\
                         [1, 1, self.width - 2, self.height - 2], 1)
        pygame.draw.rect(self.surf, (0, 0, 0),\
                         [5, self.height - 9, self.width - 10, 2], 0)
        pygame.draw.rect(self.surf, (255, 255, 255),\
                         [5, self.height - 10, self.width - 10, 4], 1)
        #Slider text
        self.txt_surf, self.txt_rect = pygame.freetype.\
                                       Font('NotoSansMono-Regular.ttf', 10).\
                                       render(self.name + f'{self.val:1.2}',\
                                              (255, 255, 255))
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
        pos = (5 + \
               int((self.val - self.min_v) / (self.max_v - self.min_v) *\
                   (self.width - 10)), self.height - 8)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)
        center = self.surf.get_rect().center
        self.txt_rect = self.txt_surf.get_rect(center=(center[0],\
                                                       center[1] - 4))
        surf.blit(self.txt_surf, self.txt_rect)
        self.window.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """
        Moves the slider handle and updates text.
        """
        self.txt_surf, _ = pygame.freetype.\
                           Font('NotoSansMono-Regular.ttf', 10).\
                           render(self.name + f'{self.val:1.4}',\
                                  (255, 255, 255))
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - 5) /\
                   (self.width - 10) * (self.max_v - self.min_v) + self.min_v
        self.val = np.clip(self.val, self.min_v, self.max_v)

def reactdiffuse():
    def update_arrays():
        """
        Vectorized implementation of the Gray-Scott algorithm.

        Read more here:
        https://www.algosome.com/articles/reaction-diffusion-gray-scott.html
        """
        weights = np.array([[.05, .2, .05],\
                            [0.2, -1, 0.2],\
                            [.05, .2, .05]])

        cv2.filter2D(arrays.A, ddepth=-1,\
                     kernel=params.diffusion_of_A * weights,\
                     dst=arrays.laplace_A, borderType=2)
        cv2.filter2D(arrays.B, ddepth=-1,\
                     kernel=params.diffusion_of_B * weights,\
                     dst=arrays.laplace_B, borderType=2)
        #Alternatively, use scipy convolution:
        #nd.convolve(arrays.A, weights, mode='wrap', output=arrays.laplace_A)
        #nd.convolve(arrays.B, weights, mode='wrap', output=arrays.laplace_B)

        #React chance is used in both equations
        np.multiply(arrays.B, arrays.B, out=arrays.react_chance)
        np.multiply(arrays.A, arrays.react_chance, out=arrays.react_chance)
        #First Equation done in place
        np.multiply(arrays.A, 1 - params.feed, out=arrays.new_A)
        np.add(arrays.new_A, arrays.laplace_A, out=arrays.new_A)
        np.subtract(arrays.new_A, arrays.react_chance, out=arrays.new_A)
        np.add(arrays.new_A, params.feed, out=arrays.new_A)
        #Second Equation done in place
        np.multiply(arrays.B, 1 - params.kill - params.feed, out=arrays.new_B)
        np.add(arrays.new_B, arrays.laplace_B, out=arrays.new_B)
        np.add(arrays.new_B, arrays.react_chance, out=arrays.new_B)
        #Clip arrays
        np.clip(arrays.new_A, 0, 1, out=arrays.A)
        np.clip(arrays.new_B, 0, 1, out=arrays.B)

    def color():
        """
        Currently we are scaling the difference, B-A, to be between 0 and 255,
        then we stack that value 3 times to create a (R,G,B) color.
        """
        np.subtract(arrays.B, arrays.A, out=arrays.difference)
        np.add(arrays.difference, 1, out=arrays.difference)
        np.multiply(arrays.difference, 127.5, out=arrays.difference)
        return np.dstack([arrays.difference]*3)

    def get_user_input():
        nonlocal running
        nonlocal hide_sliders
        nonlocal mouse_down
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_ESCAPE:
                    hide_sliders = not hide_sliders
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
                if event.button == pygame.BUTTON_LEFT:
                    pos = pygame.mouse.get_pos()
                    for slider in sliders:
                        if slider.button_rect.collidepoint(pos):
                            slider.hit = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False
                for slider in sliders:
                    slider.hit = False
                params.feed = sliders[0].val
                params.kill = sliders[1].val
                params.diffusion_of_A = sliders[2].val
                params.diffusion_of_B = sliders[3].val

    def add_substance():
        if not any([slider.hit for slider in sliders]):
            try:
                pos = pygame.mouse.get_pos()
                arrays.B[pos[0] - 4:pos[0] + 5,\
                         pos[1] - 4:pos[1] + 5] = drop
            except ValueError:
                #Too close to border
                pass

    def reset():
        arrays.A = np.ones(window_dim, dtype=np.float32)
        arrays.B = np.zeros(window_dim, dtype=np.float32)
        arrays.B[window_dim[0]//2 - 10: window_dim[0]//2 + 11,\
                 window_dim[1]//2 - 10: window_dim[1]//2 + 11] = 1

    def draw_sliders():
        for slider in sliders:
            if slider.hit:
                slider.move()
            slider.draw()

    #Game variables-----------------------------------------------------------
    window_dim = [500, 500]
    window = pygame.display.set_mode(window_dim)
    params = types.SimpleNamespace(diffusion_of_A=1., diffusion_of_B=.5,\
                                   feed=.01624, kill=.04465)
    arrays = types.SimpleNamespace()
    arrays.laplace_A = np.zeros(window_dim, dtype=np.float32)
    arrays.laplace_B = np.zeros(window_dim, dtype=np.float32)
    arrays.react_chance = np.zeros(window_dim, dtype=np.float32)
    arrays.new_A = np.zeros(window_dim, dtype=np.float32)
    arrays.new_B = np.zeros(window_dim, dtype=np.float32)
    arrays.difference = np.zeros(window_dim, dtype=np.float32)
    reset()
    sliders = [(Slider("feed = ", params.feed, [.001, .08], [20, 20], window))]
    sliders.append(Slider("kill = ", params.kill,\
                          [.01, .073], [20, 52], window))
    sliders.append(Slider("diffusion of A = ", params.diffusion_of_A,\
                           [.8, 1.2], [20, 84], window))
    sliders.append(Slider("diffusion of B = ", params.diffusion_of_B,\
                           [.4, .6], [20, 116], window))
    hide_sliders = False
    mouse_down = False

    drop = np.array([[0., 0., 1., 1., 1., 1., 1., 0., 0.],\
                     [0., 1., 1., 1., 1., 1., 1., 1., 0.],\
                     [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                     [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                     [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                     [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                     [1., 1., 1., 1., 1., 1., 1., 1., 1.],\
                     [0., 1., 1., 1., 1., 1., 1., 1., 0.],\
                     [0., 0., 1., 1., 1., 1., 1., 0., 0.],])
    #Main Loop----------------------------------------------------------------
    running = True
    while running:
        update_arrays()
        update_arrays()
        pygame.surfarray.blit_array(window, color())
        if not hide_sliders:
            draw_sliders()
        if mouse_down:
            add_substance()
        pygame.display.update()
        get_user_input()

def main():
    pygame.init()
    pygame.display.set_caption('reaction-diffusion')
    reactdiffuse()
    pygame.quit()

if __name__ == "__main__":
    main()
