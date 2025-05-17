import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

'''
BG_pivot is the same shape as the input image but with single channel, all pixels have value 1.
'''

class MOG():
    def __init__(self,height=360, width=640, number_of_gaussians=None, background_thresh=None, lr=None):
        self.number_of_gaussians = number_of_gaussians
        self.background_thresh = background_thresh
        self.lr = lr
        self.height = height
        self.width = width
        self.mus = np.zeros((self.height,self.width, self.number_of_gaussians,3)) ## assuming using color frames
        self.sigmaSQs = np.zeros((self.height, self.width, self.number_of_gaussians)) ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas = np.zeros((self.height, self.width, self.number_of_gaussians))
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i,j]=np.array([[122, 122, 122]]*self.number_of_gaussians) ##assuming a [0,255] color channel
                self.sigmaSQs[i,j]=[36.0]*self.number_of_gaussians
                self.omegas[i,j]=[1.0/self.number_of_gaussians]*self.number_of_gaussians
                
    def updateParam(self, img, BG_pivot):
        pass
        
subtractor=MOG()
frame = cv2.imread('person.jpg')
labels = subtractor.updateParam(frame, np.ones(frame.shape[:2]))
