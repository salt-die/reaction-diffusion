#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pygame implementation of reaction-diffusion.

'esc' to hide/show the sliders
click to add more chemical
'r' to reset
"""
import numpy as np
import pygame
import pygame.freetype
import scipy.ndimage as nd
import types

class Slider():
    """
    Slider code lifted from
    https://www.dreamincode.net/forums/topic/401541-buttons-and-sliders-in-pygame/
    """
    def __init__(self, name, val, min_v, max_v, xpos, ypos, window):
        self.val = val
        self.min_v = min_v
        self.max_v = max_v
        self.xpos = xpos
        self.ypos = ypos
        self.surf = pygame.surface.Surface((90, 30))
        self.surf.set_alpha(200)
        self.name = name
        self.hit = False
        self.txt_surf, _ = pygame.freetype.\
                           Font('NotoSansMono-Regular.ttf', 10).\
                           render(self.name + f'{self.val:1.2}',\
                                  (255, 255, 255))
        self.window = window
        self.surf.fill((100, 100, 100))
        pygame.draw.rect(self.surf, (255, 255, 255), [5, 20, 80, 2], 0)
        self.button_surf = pygame.surface.Surface((20, 20))
        self.button_surf.fill((1, 1, 1))
        self.button_surf.set_colorkey((1, 1, 1))
        pygame.draw.circle(self.button_surf, (0, 0, 0), (10, 10), 3)

    def draw(self):
        surf = self.surf.copy()
        pos = (5 + int((self.val-self.min_v) / (self.max_v-self.min_v) * 80), 21)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)
        self.window.blit(surf, (self.xpos, self.ypos))
        self.window.blit(self.txt_surf, (self.xpos + 15, self.ypos + 5))

    def move(self):
        self.txt_surf, _ = pygame.freetype.\
                           Font('NotoSansMono-Regular.ttf', 10).\
                           render(self.name + f'{self.val:1.3}',\
                                  (255, 255, 255))
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - 10) / 80 *\
                   (self.max_v - self.min_v) + self.min_v
        if self.val < self.min_v:
            self.val = self.min_v
        if self.val > self.max_v:
            self.val = self.max_v

def reactdiffuse():
    def update_arrays():
        weights = np.array([[.05, .2, .05],\
                            [0.2, -1, 0.2],\
                            [.05, .2, .05]])

        new_A = variables.A_array + \
                variables.diffusion_of_A *\
                nd.convolve(variables.A_array, weights, mode='wrap') -\
                variables.A_array * variables.B_array**2 + \
                variables.feed * (1 - variables.A_array)

        new_B = variables.B_array + \
                variables.diffusion_of_B *\
                nd.convolve(variables.B_array, weights, mode='wrap') +\
                variables.A_array * variables.B_array**2 - \
                (variables.kill + variables.feed) * variables.B_array

        return np.clip(new_A, 0, 1), np.clip(new_B, 0, 1)

    def color():
        difference = ((variables.A_array - variables.B_array + 1) * 127.5).astype(int)
        return np.dstack([difference for i in range(3)])

    def get_user_input():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                variables.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_ESCAPE:
                    variables.hide_sliders = not variables.hide_sliders
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    pos = pygame.mouse.get_pos()
                    for s in sliders:
                        if s.button_rect.collidepoint(pos):
                            s.hit = True
                    if not any([s.hit for s in sliders]):
                        try:
                            variables.B_array[pos[0] - 4:pos[0] + 5,\
                                              pos[1] - 4:pos[1] + 5] = 1
                        except ValueError:
                            #print("Poked too close to border.")
                            pass
            elif event.type == pygame.MOUSEBUTTONUP:
                for s in sliders:
                    s.hit = False
                variables.kill = kill_slider.val
                variables.feed = feed_slider.val

    def reset():
        variables.A_array = np.ones(window_dim, dtype=np.float32)
        variables.B_array = np.zeros(window_dim, dtype=np.float32)
        variables.B_array[window_dim[0]//2 - 10: window_dim[0]//2 + 11,\
                          window_dim[1]//2 - 10: window_dim[1]//2 + 11] = 1

    def draw_sliders():
        for s in sliders:
            if s.hit:
                s.move()
            s.draw()

    #Game variables-----------------------------------------------------------
    window_dim = [500, 500]
    window = pygame.display.set_mode(window_dim)
    variables = types.SimpleNamespace(A_array=0, B_array=0,\
                                      diffusion_of_A=1., diffusion_of_B=.5,\
                                      feed = .0550, kill= .0620)
    reset()
    feed_slider = Slider("f = ", .0550, .01, .1, 20, 20, window)
    kill_slider = Slider("k = ", .0620, .045, .07, 20, 55, window)
    diff_A_slider = Slider("D_A = ", 1., .8, 1.2, 20, 90, window)
    diff_B_slider = Slider("D_B = ", .5, .4, .6, 20, 125, window)
    sliders = [feed_slider, kill_slider, diff_A_slider, diff_B_slider]
    variables.hide_sliders = False
    #Main Loop----------------------------------------------------------------
    variables.running = True
    while variables.running:
        variables.A_array, variables.B_array = update_arrays()
        pygame.surfarray.blit_array(window, color())
        if not variables.hide_sliders:
            draw_sliders()
        get_user_input()
        pygame.display.update()

def main():
    """
    Starts reaction. Ends the reaction.
    """
    pygame.init()
    pygame.display.set_caption('reaction diffusion')
    reactdiffuse()
    pygame.quit()

if __name__ == "__main__":
    main()
