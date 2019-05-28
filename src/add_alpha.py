'''
Created on Apr 5, 2019

@author: markchristopher
'''

import cv2
import argparse
import skimage.io
import numpy as np

if __name__ == '__main__':
    '''
    Add alpha channel to given image. Alpha channel should be supplied as grayscale image.
    
    '''
    
    clargs = argparse.ArgumentParser()
    
    # Required arguments
    clargs.add_argument(
        'image_path',
        help = 'Path to image file.')
    clargs.add_argument(
        'alpha_path',
        help = 'Path to grayscale image that specifies alpha channel.')
    
    # Optional arguments
    clargs.add_argument(
        '-o', '--out_path',
        default = 'image.png',
        help = "Path to store output.")
    clargs.add_argument(
        '-s', '--scale',
        type = float,
        default = 1.0,
        help = "Scale alpha channel by given value.")
    
    args = clargs.parse_args()
    
    rgb = skimage.io.imread(args.image_path)
    alpha = skimage.io.imread(args.alpha_path)
    
    if len(alpha.shape) > 2:
        alpha = alpha[:, :, 0]
    
    alpha = (args.scale * alpha).astype(np.uint8)
    
    trans = np.dstack((rgb, alpha))
    
    skimage.io.imsave(args.out_path, trans)
    