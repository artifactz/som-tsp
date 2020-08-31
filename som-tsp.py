#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame as pg
import pygame.gfxdraw
import numpy as np
import sys, math
import subprocess


SCREEEN_WIDTH = 1080
SCREEEN_HEIGHT = 1080


def get_exp_function(start_value, end_value, start_def, end_def):
    '''creates an exponential function that returns `start_value` > 0 at `start_def` and `end_value` > 0 at `end_def`'''
    scale = 1. / (end_def - start_def)
    return lambda x: math.exp((x - start_def) * scale * math.log(end_value) + (1 - (x - start_def) * scale) * math.log(start_value))


class FfmpegVideoWriter:
    def __init__(self, filename, width, height, fps, input_pixfmt='rgba', video_filter=None):
        args = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-pixel_format', input_pixfmt,
            '-video_size', '{}x{}'.format(width, height),
            '-framerate', '{}'.format(fps),
            '-i', '-',
        ]
        if video_filter is not None:
            args += [
                '-filter:v', video_filter,
            ]
        args += [
            '-vc', 'libx264',
            '-crf', '20',
            '-y',
            filename
        ]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE)

    def encode_image(self, img):
        self.p.stdin.write(img)

    def close(self):
        self.p.terminate()


class Screen(object):
    def __init__(self, width, height, state, filename=None):
        ''':param state: a SOMState object
           :param filename: video filename (optional)'''
        self.width = width
        self.height = height
        self.state = state
        self.frame = 1
        self.filename = filename

        # init pygame
        pg.init()
        pg.display.set_caption('som-tsp')
        self.surface = pg.display.set_mode((self.width, self.height))

        # define colors
        self.col_bg = pg.Color(25, 27, 25)
        self.col_city = pg.Color(255, 255, 255)
        self.col_path = pg.Color(216, 151, 31)

        if self.filename is not None:
            self.video_writer = FfmpegVideoWriter(filename, width, height, 30, 'bgra')

    def draw_aaline(self, color, startpos, endpos, width):
        '''draws an anti-aliased line with given width'''
        (x1, y1), (x2, y2) = startpos, endpos
        # angle orthogonal to line
        a = math.atan2(y2 - y1, x2 - x1) + math.pi / 2.
        # define thicc line as a polygon
        d = width / 2.
        points = [(x1 + math.cos(a) * d, y1 + math.sin(a) * d),
            (x1 - math.cos(a) * d, y1 - math.sin(a) * d),
            (x2 - math.cos(a) * d, y2 - math.sin(a) * d),
            (x2 + math.cos(a) * d, y2 + math.sin(a) * d)]
        # draw
        pg.gfxdraw.aapolygon(self.surface, points, color)
        pg.gfxdraw.filled_polygon(self.surface, points, color)

    def run(self):
        '''main GUI loop'''

        clock = pg.time.Clock()

        while True:
            # events
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_q or event.key == pg.K_ESCAPE:
                        sys.exit()
                elif event.type == pg.QUIT:
                    sys.exit()

            # drawing
            self.surface.fill(self.col_bg)

            # path
            k = self.state.k
            for map_idx in range(k):
                x1, y1 = self.state.map[:, map_idx]
                x2, y2 = self.state.map[:, (map_idx + 1) % k]
                self.draw_aaline(self.col_path, (x1, y1), (x2, y2), 3)

            # cities
            for city_idx in range(self.state.n):
                x, y = map(int, self.state.cities[:, city_idx])
                pg.gfxdraw.aacircle(self.surface, x, y, 5, self.col_city)
                pg.gfxdraw.filled_circle(self.surface, x, y, 5, self.col_city)

            # video
            if self.filename is not None:
                self.video_writer.encode_image(self.surface.get_buffer().raw)

            pg.display.flip()
            clock.tick(30)  # limit to 30 fps
            self.state.step()
            self.frame += 1

            if self.state.is_finished():
                if self.state.next_state:
                    self.state = self.state.next_state
                else:
                    break

        self.video_writer.close()


class SOMState(object):
    def __init__(self, iterations, cities, next_state=None):
        self.cities = cities
        self.n = cities.shape[1]
        self.k = self.n ** 2
        self.map = self.cities[:, np.random.randint(0, self.n, self.k)] + np.random.randn(2, self.k) * 20

        self.iterations = iterations
        self.iteration = 1
        self.sigma = get_exp_function(self.n * 5, 1., 0, self.iterations)  # "brush" width
        self.rho = get_exp_function(0.2, 0.1, 0, self.iterations)  # "brush" intensity

        self.next_state = next_state

    def is_finished(self):
        return self.iteration > self.iterations

    def get_proximity(self, p, q):
        '''gaussian neighborhood function'''
        d = abs(p - q)
        d = min(d, self.k - d)
        return math.e ** (-(d ** 2) / (2 * self.sigma(self.iteration) ** 2))

    def step(self):
        if self.is_finished():
            return

        # move closest item in solution and its neighborhood towards each city
        for city_idx in range(self.n):
            city = self.cities[:, city_idx]
            closest_map_idx = np.argmin((np.abs(self.map - city[:, np.newaxis]) ** 2).sum(axis=0) ** 0.5)  # nearest solution item
            for map_idx in range(self.k):
                self.map[:, map_idx] += self.rho(self.iteration) * self.get_proximity(closest_map_idx, map_idx) * (city - self.map[:, map_idx])

        self.iteration += 1


# test case types:

def get_uniform_cities(width, height, n):
    margin_h = width / 15.
    margin_v = height / 15.
    return np.vstack((np.random.randint(0, width - margin_h * 2, n) + margin_h, np.random.randint(0, height - margin_v * 2, n) + margin_v)).astype(np.float64)

def get_grid_cities(width, height, cols, rows):
    cities = np.zeros((2, cols * rows))
    k = 0
    for i in range(cols):
        x = width / (cols + 1) * (i + 1)
        for j in range(rows):
            y = height / (rows + 1) * (j + 1)
            cities[:, k] = [x, y]
            k += 1
    return cities

def get_circle_cities(width, height, n1, n2):
    cities = np.zeros((2, n1 + n2))
    r1 = min(width, height) / 16. * 7.
    r2 = r1 / 2.
    for i in range(n1):
        cities[:, i] = [r1 * math.cos(2. * math.pi * i / n1), r1 * math.sin(2. * math.pi * i / n1)]
    for i in range(n2):
        cities[:, n1 + i] = [r2 * math.cos(2. * math.pi * i / n2), r2 * math.sin(2. * math.pi * i / n2)]
    center = np.asarray([width / 2., height / 2.])
    return cities + center[:, np.newaxis]

def get_cluster_cities(width, height, n, centers):
    cities = np.zeros((2, n * len(centers)))
    std = min(width, height) / 11.5
    i = 0
    for k, (x, y) in enumerate(centers):
        center = np.asarray([x, y])
        cities[:, k*n:(k+1)*n] = np.random.randn(2, n) * std + center[:, np.newaxis]
    return cities

def get_halves_cities(width, height, n):
    cities = np.zeros((2, n * 2))
    margin_h = width / 15.
    margin_v = height / 15.
    for i in range(n):
        cities[:, 2*i] = [np.random.randint(0, width / 4. - margin_h) + margin_h, np.random.randint(0, height - margin_v * 2) + margin_v]
        cities[:, 2*i+1] = [np.random.randint(0, width / 4. - margin_h) + width / 4. * 3., np.random.randint(0, height - margin_v * 2) + margin_v]
    return cities


if __name__ == '__main__':
    # build test cases
    np.random.seed(124)
    cities_g1 = get_grid_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 3, 3)
    cities_g2 = get_grid_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 4, 4)
    cities_g3 = get_grid_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 5, 5)
    cities_u1 = get_uniform_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 20)
    cities_u2 = get_uniform_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 40)
    cities_c1 = get_circle_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 11, 5)
    cities_k1 = get_cluster_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 12, ((300, 300), (SCREEEN_WIDTH - 300, SCREEEN_HEIGHT - 300)))
    cities_k2 = get_cluster_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 9, ((266, 266), (SCREEEN_WIDTH - 266, 266), (SCREEEN_WIDTH / 2., SCREEEN_HEIGHT - 266)))
    cities_h1 = get_halves_cities(SCREEEN_WIDTH, SCREEEN_HEIGHT, 15)

    # test case sequence
    state = SOMState(150, cities_g1,
            SOMState(150, cities_c1,
            SOMState(150, cities_h1,
            SOMState(150, cities_g2,
            SOMState(150, cities_k1,
            SOMState(150, cities_u1,
            SOMState(150, cities_g3,
            SOMState(150, cities_k2,
            SOMState(150, cities_u2)))))))))

    # screen = Screen(SCREEEN_WIDTH, SCREEEN_HEIGHT, state, 'som-tsp.mp4')  # video mode
    screen = Screen(SCREEEN_WIDTH, SCREEEN_HEIGHT, state)
    screen.run()
