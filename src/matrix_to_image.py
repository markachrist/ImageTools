'''
Created on Dec 31, 2018

@author: markchristopher
'''

import argparse
import numpy as np
import skimage.io
import skimage.exposure

if __name__ == '__main__':
    
    clargs = argparse.ArgumentParser()
    
    # Required arguments
    clargs.add_argument(
        'matrix_path',
        help = 'Path to npy file.')
    
    # Optional arguments
    clargs.add_argument(
        '-o', '--outpath',
        default = 'image.tif',
        help = "Path to store output.")
    
    args = clargs.parse_args()
    
    m = np.load(args.matrix_path)
    
    i = skimage.exposure.rescale_intensity(m, out_range = np.uint8)
    skimage.io.imsave(args.outpath, i.astype(np.uint8))
    