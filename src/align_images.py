'''

Performs basic registration to align one image to another.

Created on Sep 2, 2016

@author: mchristopher
'''

import imtools
import argparse
import numpy
import cv2
import skimage.io
import skimage.filters
import skimage.transform

import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    Aligns two images using basic OpenCV methods.
    
    Usage:
    
    %python align_images.y [options] <target_image> <other_image>
    
    Finds transform that aligns <other_image> to <target_image>. Resulting transformed image
    is output along with coefficients of transform used to align the images. 
    
    '''
    clargs = argparse.ArgumentParser()
     
    #Required arguments 
    clargs.add_argument(
        'target_path',
        help = 'Path to target image.')
    clargs.add_argument(
        'other_path',
        help = 'Path to image that will be aligned to the target.')
     
    #Optional arguments
    clargs.add_argument(
        '-o', '--out_path',
        default = './aligned.tif',
        help = "Path used to store output.")
    clargs.add_argument(
        '-i', '--initial',
        default = '1,0,0,0,1,0',
        help = "Initial alignment transform. Defaults to identity transform.")
    clargs.add_argument(
        '-n', '--num_its',
        type = int,
        default = 10000,
        help = "Max number of iterations in alignment search.")
    clargs.add_argument(
        '-s', '--scales',
        type = int,
        default = 1,
        help = "Number of levels in alignment scale space alignment. 1 (default) indicates " +
                "alignment of raw images and no down-sampled alignment.")
    clargs.add_argument(
        '-t', '--type',
        default = cv2.MOTION_EUCLIDEAN,
        help = "The type of transform used to align images. Options: " + 
                str(cv2.MOTION_TRANSLATION) + " (Translation), " +
                str(cv2.MOTION_EUCLIDEAN) + " (Euclidean), " +
                str(cv2.MOTION_AFFINE) + " (Affine), " +
                str(cv2.MOTION_HOMOGRAPHY) + " (Homographic)")
    args = clargs.parse_args()
    
    # Get input images    
    target = skimage.io.imread(args.target_path)
    other = skimage.io.imread(args.other_path)
    
    # Type of transform
    tform_type = cv2.MOTION_EUCLIDEAN
    tform_type = cv2.MOTION_AFFINE
    
    # Initial transform
#     tform = numpy.eye(3, dtype = numpy.float32)
    tform = numpy.reshape(numpy.fromstring(args.initial.strip(), sep = ","), (2, 3))
    tform = tform.astype(numpy.float32)
    
    #Termination criteria
    max_its = args.num_its
    term_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_its, term_eps)
    
    target = skimage.filters.sobel(target)
    other = skimage.filters.sobel(other)
    
    # Run the registration
    had_last_it = False
    for s in range(args.scales, 0, -1):
        
        scale = 2**s
        
        if scale > 1:
            cur_target = skimage.transform.pyramid_reduce(target, downscale = scale, mode = 'reflect').astype(numpy.float32)
            cur_other = skimage.transform.pyramid_reduce(other, downscale = scale, mode = 'reflect').astype(numpy.float32)
        else:
            cur_target = target
            cur_other = other
        
        # Scale up transform
        if had_last_it:
            tform[0, 2] = 2.0 * tform[0, 2]
            tform[1, 2] = 2.0 * tform[1, 2]
        
        (rcode, tform) = cv2.findTransformECC(cur_target, cur_other, tform, tform_type, criteria)
        had_last_it = True
        
#     (rcode, tform) = cv2.findTransformECC(target, other, tform, tform_type, criteria)
    
    # Transform image to target
    tflags = cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(other, tform, (target.shape[1], target.shape[0]), flags = tflags)
    
    # Output
    skimage.io.imsave(args.out_path, aligned)
    
    tup = (tform[0, 0], tform[0, 1], tform[0, 2], tform[1, 0], tform[1, 1], tform[1, 2])
    message = "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f" % tup
    print message
    
    