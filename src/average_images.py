'''
Created on Dec 7, 2016

@author: markchristopher
'''

import argparse

import numpy
import skimage.io

if __name__ == '__main__':
    '''
    Averages set of images.
    
    Usage:
    
    (1) %python average_images.y [options] <image 1> [<image 2> <image 3> ...]
    (2) %python average_images.y [options] -f <path to list>
    
    Generates average image from the given set of input images. Input images should all have same
    width, height, pixel type. 
    
    '''
    clargs = argparse.ArgumentParser()
     
    #Required arguments 
    clargs.add_argument(
        'image_list',
        default = [],
        nargs = '*',
        help = 'Set of images to average.')
     
    #Optional arguments
    clargs.add_argument(
        '-o', '--out_path',
        default = './average.tif',
        help = "Path used to store output.")
    clargs.add_argument(
        '-f', '--file_list',
        default = '',
        help = "Path to file ")
    args = clargs.parse_args()
    
    list_to_use = None
    
    # Get input list
    if len(args.file_list) > 0:
        f = open(args.file_list, 'r')
        list_to_use = [x.strip() for x in f.readlines()]
        f.close()
    elif len(args.image_list) > 0:
        list_to_use = args.image_list
        
    # No input
    if not list_to_use or len(list_to_use) == 0:
        clargs.error('No input images provided.')
        
    # Average
    avg = numpy.empty(0)
    n = 0
    for i in list_to_use:
        image = skimage.io.imread(i)
        
        if avg.size == 0:
            avg = numpy.zeros(image.shape)
        
        avg += image
        n += 1
    
    avg /= n
    
    # Save output
    skimage.io.imsave(args.out_path, avg.astype(image.dtype))
    
    