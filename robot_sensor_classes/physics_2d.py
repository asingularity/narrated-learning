import random
import cv2
import numpy as np
import pickle


random.seed(0)


class Physics2DSensor(object):
    def __init__(self, params):

        self.image_dim = params['image_dim']
        self.return_type = params['return_type']

        assert params['return_type'] is np.float32 or params['return_type'] is np.float64
        self.return_type = params['return_type']

    def read_input(self):


        im_size = self.image_dim
        circles_diameter_prop_im = [0.1, 0.2, 0.4]

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
