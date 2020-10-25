import random
import cv2
import numpy as np
import pickle
import numpy as np
import cv2
import pymunk
import pymunk.util
from pymunk import Vec2d
import math, sys, random


random.seed(0)


class Physics2DSensor(object):
    def __init__(self, params):

        self.image_dim = params['image_dim']
        self.return_type = params['return_type']

        print('image dim:', self.image_dim)

        if 'take_subimage_factor' in params:
            self.take_subimage_factor = params['take_subimage_factor']
        else:
            self.take_subimage_factor = None

        self.orig_image_dim = self.image_dim

        if self.take_subimage_factor is not None:
            self.image_dim = int(self.image_dim * self.take_subimage_factor)

        factor = self.image_dim / 800.0

        # TODO make this a demo parameter
        self.num_balls_max = 3
        # TODO should work when ball smaller as well
        self.ball_radius = 2 * 50 * factor
        self.init_velocity_scale = 5 * 30 * factor
        self.min_steps_between_balls = 0  # 20
        self.mass = 10  #* factor
        self.gravity = -90 * 10
        assert params['return_type'] is np.float32 or params['return_type'] is np.float64
        self.return_type = params['return_type']

        space = pymunk.Space()
        space.gravity = (0.0, self.gravity)

        static_body = space.static_body
        static_lines = [pymunk.Segment(static_body, (0.0, 0.0), (self.image_dim, 0.0), 0.0)]  # ,
        # pymunk.Segment(static_body, (0.0, 400.0), (0.0, 0.0), 0.0),
        # pymunk.Segment(static_body, (400.0, 400.0), (400.0, 0.0), 0.0)]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.0
        space.add(static_lines)

        self.balls = []
        self.space = space

        self.steps_to_next_ball = 10

    def read_input(self):
        self.steps_to_next_ball -= 1
        if len(self.balls) < self.num_balls_max and self.steps_to_next_ball <= 0:
            self.steps_to_next_ball = self.min_steps_between_balls

            mass = self.mass
            radius = self.ball_radius
            inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            body = pymunk.Body(mass, inertia)
            x = random.randint(1, self.image_dim - 1)
            body.position = x, self.image_dim - 1

            rand_vel = 1.0 + random.random()
            if random.random() < 0.5:
                rand_vel = -rand_vel

            body.velocity = self.init_velocity_scale * rand_vel, 0
            shape = pymunk.Circle(body, radius, Vec2d(0, 0))
            shape.elasticity = 0.95
            self.space.add(body, shape)
            self.balls.append(shape)

        balls_to_remove = []

        im = 0.3 * np.ones((self.image_dim, self.image_dim), self.return_type)

        for ball in self.balls:
            if ball.body.position.y < 0:
                balls_to_remove.append(ball)
            else:
                cv2.circle(img=im, center=(int(ball.body.position.x), self.image_dim - int(ball.body.position.y)), radius=int(ball.radius), color=0.8, thickness=-1)

        for ball in balls_to_remove:
            self.space.remove(ball, ball.body)
            self.balls.remove(ball)

        self.space.step(1 / 100.0)

        if self.take_subimage_factor is not None:
            #im = im[im.shape[0] - self.orig_image_dim::, im.shape[0] - self.orig_image_dim::]
            # take center
            start_r = int(self.orig_image_dim / 2)
            start_c = start_r
            dim = self.orig_image_dim
            im = im[start_r:start_r + dim, start_c:start_c + dim]

        assert im.shape[0] == self.orig_image_dim and im.shape[1] == self.orig_image_dim

        return im


    def read_input_old(self):

        im_size = self.image_dim
        circles_diameter_prop_im = [0.1, 0.2, 0.4]
        #circles_diameter_prop_im = [0.2, 0.4, 0.8]

        im = 50 * np.ones((im_size, im_size), np.uint8)
        for circ_diam_prop in circles_diameter_prop_im:
            rad = int((circ_diam_prop * im_size) / 2.0)
            cv2.circle(img=im, center=(random.randint(0, im_size), random.randint(0, im_size)), radius=rad, color=200, thickness=-1)

        im = im.astype(self.return_type) * 1.0/255.0

        return im



def main():
    sensor = Physics2DSensor(params={
        'image_dim': 128,
        'return_type': np.float32
    })

    while True:
        cv2.imshow('im', sensor.read_input())
        cv2.waitKey(300)



if __name__ == '__main__':
    main()
