'''
Created on Dec 31, 2018

@author: markchristopher
'''

import cv2
import os.path
import argparse
import skimage.io
import numpy as np

import skimage.color

if __name__ == '__main__':
    
    clargs = argparse.ArgumentParser()
    
    # Required arguments
    clargs.add_argument(
        'image_path',
        help = 'Path to image file.')
    
    # Optional arguments
    clargs.add_argument(
        '-o', '--out_path',
        default = 'image.tif',
        help = "Path to store output.")
    clargs.add_argument(
        '-u', '--under_path',
        default = None,
        help = "Path to image that input will be laid over.")
    
    args = clargs.parse_args()
    
    i = skimage.io.imread(args.image_path)
    
    if len(i.shape) == 2:
        i = skimage.color.gray2rgb(i)
#     output = cv2.applyColorMap(i, cv2.COLORMAP_RAINBOW)
#     skimage.io.imsave(args.out_path, cv2.applyColorMap(output.astype(np.uint8), cv2.COLORMAP_RAINBOW))
    
    under = np.zeros(i.shape)
    alpha  = i.astype(np.float32) / 255.0
    alpha *= 1.0
#     blend = (1.0 - alpha) * (255.0) * under + alpha * cv2.applyColorMap(i.astype(np.uint8), cv2.COLORMAP_RAINBOW)
    blend = (1.0 - alpha) * (255.0) * under + alpha * cv2.applyColorMap(i.astype(np.uint8), cv2.COLORMAP_PARULA)
    skimage.io.imsave(args.out_path, blend.astype(np.uint8))
    
    if not args.under_path == None:
        basename = os.path.basename(os.path.splitext(args.out_path)[0])
        under = skimage.io.imread(args.under_path).astype(np.float32) / 255.0
        alpha  = i.astype(np.float32) / 255.0
        alpha *= 1.0
        blend = (1.0 - alpha) * (255.0) * under + alpha * cv2.applyColorMap(i.astype(np.uint8), cv2.COLORMAP_RAINBOW)
        
        path = os.path.join(os.path.dirname(args.out_path), basename + '-over.tif')
        skimage.io.imsave(path, blend.astype(np.uint8))
    
    