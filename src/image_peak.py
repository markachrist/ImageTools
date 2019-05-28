'''
Created on Apr 1, 2019

@author: markchristopher
'''

import argparse
import numpy as np

import skimage.io
import skimage.color

from scipy import signal

if __name__ == '__main__':
    '''
    '''
    
    clargs = argparse.ArgumentParser()
     
    #Required arguments 
    clargs.add_argument(
        'image_path',
        help = 'Path to image file.')
     
    #Optional arguments
    clargs.add_argument(
        '-k', '--kernel_path',
        default = None,
        help = "Path to file cantaining optional kernel.")
    args = clargs.parse_args()
    
    image = skimage.io.imread(args.image_path)
    
    if len(image.shape) == 3:
        image = skimage.color.rgb2gray(image)
        pass
    
    if not args.kernel_path == None:
        kernel = skimage.io.imread(args.kernel_path)
        image = signal.fftconvolve(image, kernel, 'same')
        
    idx = np.argmax(image)
    print '%d,%d' % ((idx / image.shape[1]), (idx % image.shape[1]))   
    
    