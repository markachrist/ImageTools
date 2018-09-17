'''
Created on Sep 19, 2016

@author: mchristopher
'''

import cv2
import numpy
import skimage.io

import argparse

if __name__ == '__main__':
    clargs = argparse.ArgumentParser()
     
    #Required arguments 
    clargs.add_argument(
        'image_path',
        help = 'Path to input image.')
     
    #Optional arguments
    clargs.add_argument(
        '-o', '--out_path',
        default = './transformed.tif',
        help = "Path used to store output.")
    clargs.add_argument(
        '-t', '--transform',
        action = 'append',
        default = [],
        help = "Transform to apply to the image. Should be comma-separated list of affine matrix coefficients. " +
                "For the transform [[a, b, tx], [c, d, ty], [0, 0, 1]], the argument: \"a,b,tx,c,d,ty\" should be given. " +
                "Multiple transforms may be given and they are pre-multiplied before being applied.")
    clargs.add_argument(
        '-f', '--file',
        default = "",
        help = "Path to a file specifying transform(s) to apply. Each transform should be defined on a single line. " + 
                "See description of -t option for details on defining transforms. Overrides -t.")
    
    args = clargs.parse_args()
    
    image = skimage.io.imread(args.image_path)
    
    tforms = []
    
    # Read from file
    if len(args.file) > 0:
        f = open(args.file, 'r')
        tforms = f.readlines()
        f.close()
    # Read from cl
    elif len(args.transform) > 0:
        tforms = args.transform
    
    # Nothing to do
    if len(tforms) == 0:
        print "No transform specified. No action."
        exit()
    
    first = True
    for t in tforms:
        
        cur = numpy.eye(3)
        cur[0:2, :] = numpy.reshape(numpy.fromstring(t.strip(), sep = ","), (2, 3))
        
#         print "Tform: "
#         print cur
        
        if first:
            tform = cur
            first = False
            
#             det = tform[0, 0] * tform[1, 1] - tform[0, 1] * tform[1, 0]
#             
#             inverse = numpy.copy(tform)
#             
#             inverse[1, 1] = tform[0, 0] / det
#             inverse[0, 0] = tform[1, 1] / det
#             inverse[1, 0] = -tform[0, 1] / det
#             inverse[0, 1] = -tform[1, 0] / det
#             
#             inverse[0, 2] = -tform[0, 2]
#             inverse[1, 2] = -tform[1, 2]
#             
#             tform = inverse
            
        else:
            
#             det = cur[0, 0] * cur[1, 1] - cur[0, 1] * cur[1, 0]
#             
#             inverse = numpy.copy(cur)
#             
#             inverse[1, 1] = cur[0, 0] / det
#             inverse[0, 0] = cur[1, 1] / det
#             inverse[1, 0] = -cur[0, 1] / det
#             inverse[0, 1] = -cur[1, 0] / det
#             
#             inverse[0, 2] = - cur[0, 0] * cur[0, 2] - cur[1, 0] * cur[1, 2]
#             inverse[1, 2] = - cur[1, 1] * cur[1, 2] + cur[0, 1] * cur[0, 2]
#             
#             cur = inverse
            inverse = cv2.invertAffineTransform(cur[0:2,])
            cur[0:2,] = inverse
            tform = numpy.dot(tform, cur)
        
    
    flags = cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP
    flags = cv2.INTER_CUBIC
    aligned = cv2.warpAffine(image, tform[0:2, :], (image.shape[1], image.shape[0]), flags = flags)
    
    print "Saving output to: " + args.out_path
    skimage.io.imsave(args.out_path, aligned)
    
    
    