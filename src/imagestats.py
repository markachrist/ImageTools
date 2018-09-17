'''
Created on Jul 31, 2016

@author: mchristopher
'''

import argparse
import numpy
import skimage.io
import imtools

if __name__ == '__main__':
    '''
    Computes basic statistics about a set of images.
    
    Outputs a images showing the mean and variance at each pixel location across the set of images.
    
    '''
    
    clargs = argparse.ArgumentParser()
     
    #Required arguments 
    clargs.add_argument(
        'image_list',
        help = 'Path to the directory containing SS-OCT data.')
     
    #Optional arguments
    clargs.add_argument(
        '-o', '--out_path',
        default = '.',
        help = "Output path. Should be a directory or an .npz file. " \
         "If a directory, mean and variance images are saved there. " \
         "If an .npz file, numpy array data is stored.")
    clargs.add_argument(
        '-p', '--prefix',
        default = '',
        help = "Prefix to added to each output filename.")
     
    args = clargs.parse_args()
    
    imageFile = open(args.image_list)
    n = 0

    # Two-pass mean/variance calculation
    # Two reads of each image, but avoids memory issues when handling large numbers of images
    for line in imageFile:
        
        curImage = skimage.io.imread(line.strip())
        
        width = curImage.shape[1]
        height = curImage.shape[0]
        
        if len(curImage.shape) < 3:
            curImage = numpy.reshape(curImage, (height, width, 1))
        
        channels = curImage.shape[2]
        
        if n == 0:
            mean = numpy.zeros((height, width, channels), dtype = numpy.float32)
            var = numpy.zeros((height, width, channels), dtype = numpy.float32)
    
        mean += curImage
        n += 1
    
    mean /= n
    
    imageFile.close()
    imageFile = open(args.image_list)
    
    for line in imageFile:
        
        curImage = skimage.io.imread(line.strip())
        
        width = curImage.shape[1]
        height = curImage.shape[0]
        
        if len(curImage.shape) < 3:
            curImage = numpy.reshape(curImage, (height, width, 1))
        
        diff = mean - curImage
        var += numpy.power(diff, 2)
        
    var /= n
    
    imageFile.close()
    
    # Output result as .npz file with two elements named 'mean' and 'var'
    if args.out_path.endswith('.npz'):
        numpy.savez(args.out_path, mean = mean, var = var)
    else:
        imtools.saveAsByte(mean, args.out_path + "/" + args.prefix + "_mean.tif")
        imtools.saveAsByte(var, args.out_path + "/" + args.prefix + "_var.tif")
    
    