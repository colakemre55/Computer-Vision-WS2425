import numpy as np
import cv2 as cv

'''
    load the face image and foreground/background parts
    image: the original image
    foreground/background: numpy arrays of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''

def read_image(filename):
    image = cv.imread(filename) / 255
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[90:350, 110:280, :] = 1
    bb_width, bb_height = 170, 260
    
    test = image * bounding_box 
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))
    background = image[bounding_box == 0].reshape((height * width - bb_width * bb_height, 3))
    return image, foreground, background
    
class GMM(object):
    
    def gaussian_scores(self, data):
        pass
    
    def estep(self, data):
        pass

    def mstep(self, data, r):
        pass

    def em_algorithm(self, data, n_iterations=10):
        pass

    def kmeans_init(self, data):
        pass

    def probability(self, data):
        pass

    def train(self, data):
        pass
        
        
image, foreground, background = read_image('person.jpg')
'''
TODO: compute p(x|w=background) for each image pixel and manipulate image such that everything below the threshold is black, display the resulting image
Hint: Slide 64
'''
# gmm_foreground, gmm_background = GMM(), GMM()
gmm_background = GMM()
