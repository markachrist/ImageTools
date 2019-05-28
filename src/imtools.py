'''
Created on Jul 19, 2016

@author: mchristopher
'''

import math
import sys
import numpy
import bisect

import time
import timeit

import numpy.fft
import scipy.signal
import scipy.ndimage
import scipy.fftpack.helper

import skimage.io
# import skimage.filters
import skimage.measure
import skimage.exposure
import skimage.morphology
import skimage.transform

# import mahotas

import cv2

# import matplotlib.pyplot
# import matplotlib.animation
# from src.apply_transform import image

def gaussian2D(sigma = (1, 1), size = None, mu = None):
    '''
    Generates 2D Gaussian image with the given means and standard deviations.
    
    Each argument should a 2-tuple specifying values in the x and y directions.
    
    @param sigma: Standard deviations in the x,y directions
    @param size: The output size in x,y directions. If None, 3 * sigma is used for each.
    @param mu: Mean x,y values. If None, the center point of the image is used.
       
    @return: Float-valued 2D Gaussian image
    '''
    
    if size == None:
        size = (numpy.around(3.0 * sigma[0]), numpy.around(3.0 * sigma[1]))
    if mu == None:
        mu = ((size[0] / 2.0 + 0.5) - 1, (size[1] / 2.0 + 0.5) - 1)
        
    gauss = numpy.zeros((int(size[0]), int(size[1])))
    
    for y in range(size[0]):
        for x in range(size[1]):
            sig = 2.0 * sigma[0] * sigma[1]
            ex = ( (y - mu[0])*(y - mu[0]) + (x - mu[1])*(x - mu[1]) ) / sig 
            gauss[y, x] = (1.0 / (sig * math.pi) ) * math.exp(-ex)
        
    return gauss

def maxGabor(image, freqs, sigmas, thetas):
    '''
    
    Computes maximum Gabor response over a range of frequencies, st. devs., and orientions.
    
    '''
    
    result = numpy.ones(image.shape) * -numpy.finfo(numpy.float64).max
     
#     for f in freqs:
#         for s in sigmas:
#             for t in thetas:
#                 skimage.filters.
#                 r, i = skimage.filters.gabor(image.astype(numpy.float32), f, theta = t, sigma_x = s, sigma_y = s)
#                 cur = numpy.absolute(r + 1j * i)
#                 cur = r
# #                 cur = cur[0]
#                  
#                 name = "freq" + str(f) + "sigma" + str(s) + "theta" + str(t)
#                 saveAsByte(cur, "/Users/mchristopher/Documents/Data/UCSD/STARFISH/ss-oct/final-fuck/" + name + ".tif")
#                  
#                 result = numpy.maximum(result, numpy.absolute(cur))
#     
#     
#     return result

def unsharpMasking(image, sigma = 10.0, amount = 2, fourier = True):
    '''
    Apply unsharp masking to enhance the edges of the given image.
    
    @param image: Image to sharpen
    @param radius: Sigma of Gaussian filters used in unsharp masking
    @param amount: Controls how much edges are enhanced (higher values -> higher contrast edges)
    @param fourier: Indicates convolution should be done in freq. domain for speed
    
    @return: Sharpened version of the image
    '''
    
    radius = int(math.ceil(2 * sigma))
    blur = cv2.getGaussianKernel(2*radius + 1, sigma) * cv2.getGaussianKernel(2*radius + 1, sigma).transpose()
    
    # High-pass filter
    sharp = numpy.zeros((2*radius + 1, 2*radius + 1))
    sharp[radius, radius] = 1.0
    sharp = sharp - blur
    
    sharp = amount * sharp
    sharp[radius, radius] = sharp[radius, radius] + 1
    
    if fourier:
        return scipy.signal.fftconvolve(image, sharp, mode = 'same')
    else:
        return scipy.ndimage.filters.correlate(image, sharp, mode = 'nearest')

def normalize(image, mn = 0.0, mx = 1.0):
    '''
    Normalizes image values to the given range by shifting/scaling pixel values.
    
    @param image: Image to be normalized
    @param mn: Min value after normalization
    @param mx: Max value after normalization
    
    @return: Normalized image with values in the range [mn, mx]
    '''
    
    if mx < mn:
        raise ValueError('mx < mn!')
    
    norm = image - image.min()
    norm = norm / norm.max()
    norm = (mx - mn) * norm + mn
    
    return norm

def homomorphic(image, boost, freq_cutoff, order, hist_cutoffs = None):
    '''
    Applies homomorphic filtering to the image.  
    
    @param image: 
    
    '''
    
    result = skimage.exposure.rescale_intensity(image, out_range = (0, 1))
    result = numpy.log(result + 0.01)
    result = numpy.fft.fft2(result)
    
    filt = boostFilter(image.shape, freq_cutoff, order, boost)
    
    result = numpy.multiply(result, filt)
    result = numpy.real(numpy.fft.ifft2(result))
    result = numpy.exp(result)
    
    if not hist_cutoffs == None:
        result = truncateHistogram(result, hist_cutoffs[0], hist_cutoffs[1])
    else:
        result = skimage.exposure.rescale_intensity(result, out_range = (0, 1))
        
    return result

def truncateHistogram(image, lo_cutoff = 0, hi_cutoff = 0):
    '''
    Adjusts pixel intensities and maps them to the range [0, 1]. The given cutoffs indicate the
    percentage of pixels that should be saturated at 0 or 1. (lo_cutoff)% of pixels will be 
    saturated at 0 and (hi_cutoff)% will be saturated at 1.
    
    @param image: Input image
    @param lo_cutoff: Percentage indicating pixels to saturate at low-end. Should be in [0, 100].
    @param hi_cutoff: Percentage indicating pixels to saturate at high-end. Should be in [0, 100].
    
    @return: Image with intensities clipped/mapped to the range [0, 1]
    '''

    norm = image - image.min()
    norm = norm / norm.max()

    m = norm.shape[0] * norm.shape[1]

    sorted_pixels = numpy.sort(image, axis = None)
    sorted_pixels = numpy.insert(sorted_pixels, (0, m), (sorted_pixels[0], sorted_pixels[m - 1]))

    x = 100 * (numpy.arange(0.5, m) / m)
    x = numpy.insert(x, (0, m), (0, 100))
    ts = numpy.interp((lo_cutoff, 100 - hi_cutoff), x, sorted_pixels)
    
    trun = skimage.exposure.rescale_intensity(image, in_range = (ts[0], ts[1]), out_range = (0, 1))
    
    return trun

def normalizeToInt(image, dtype = numpy.uint8):
    '''
    Normalizes image values to fit within the range of the given integer data type.
    
    Returns normalized image converted to the indicated data type.
    
    @param image: Image to normalize/convert
    @param dtype: An integer data type to which the image is normalized/converted
    
    @return: Image normalized/converted to given type
    '''
    
    mn = numpy.iinfo(dtype).min
    mx = numpy.iinfo(dtype).max
    
    norm = normalize(image, mn, mx)
    
    norm = norm.astype(dtype)
    
    return norm
    
def saveAsByte(image, path):
    '''
    Saves the given array as a byte-valued image. Numeric values are shifted/scaled to the
    range [0,255].
    
    @param image: Image to be saved
    @param path: String indicating output path
    
    @return: True if successfully saved, False otherwise
    '''
    
    image = image.astype(numpy.float) - image.min()
    image = (image / image.max()) * 255.0
    skimage.io.imsave(path, image.astype(numpy.uint8))
    
    return True

def loadRawImage(path, w, h, data_type = numpy.float32):
    '''
    
    Loads raw image data into array.
    
    @param path: Path to raw file.
    @param w: Width of image
    @param h: Height of image
    
    @return: Image load from raw data
    '''
    
    size = numpy.dtype(data_type).itemsize   
    f = open(path, 'rb')
    
    buf = f.read(w * h * size)
    image = numpy.frombuffer(buf, dtype = data_type)
    image = numpy.reshape(image, (h, w))
    
    f.close()
    
    return image
    

# def saveAsAnimatedGif(frames, path, delay = 500, repeat = True):
#     '''
#     
#     Create/save an animated gif from a series of images.
#     
#     @param frames: Series of images (as a list of numpy arrays) to include in the animation
#     @param path: Path to output location
#     @param delay: Interval for which each frame is displayed, in milliseconds
#     @param repeat: Indicates whether animation should played once or repeated indefinitely
#     @return: True if successfully saved, False otherwise
#     '''
#     
#     saved = True
#     
#     fig = matplotlib.pyplot.figure()
#     ax = fig.add_subplot(111)
#     ax.set_axis_off()
#  
#     ims = map(lambda x: (ax.imshow(x), ax.set_title("")), frames)
#     gif = matplotlib.animation.ArtistAnimation(fig, ims, interval = delay, repeat = repeat, blit = False)
#     
#     gif.save(path, writer = 'ffmpeg')
#     
#     return saved

def boostFilter(size, cutoff, order = 1, boost = 1.0):
    '''
    Creates a high (boost >= 1.0) or low boost filter (boost < 1.0) that can be applied to an image
    in the frequency domain.
    
    @param boost: 
    @param size: Size of the filter. Should be list/tuple with [num rows, num cols].
    @param cutoff: Cutoff frequency of the filter. Should be in [0, 0.5]
    @param order: Order of the filter. Higher values give sharper frequency drop offs. 
                  Should be integer >= 1.
    
    @return: Filter that boosts the pass band relative to other
    '''

    if boost >= 1.0:
        f = (1.0 - 1.0 / boost) * highPassFilter(size, cutoff, order) + 1.0 / boost
    else:
        f = (1.0 - boost) * lowPassFilter(size, cutoff, order) + boost
    
    return f


def lowPassFilter(size, cutoff, order):
    '''
    Creates a low pass filter that can be applied to an image in the frequency domain.
    
    @param size: Size of the filter. Should be list/tuple with [num rows, num cols].
    @param cutoff: Cutoff frequency of the filter. Should be in [0, 0.5]
    @param order: Order of the filter. Higher values give sharper frequency drop offs. 
                  Should be integer >= 1.
    
    @return: Low pass filter to apply to frequency-domain image 
    '''

    rows = size[0]
    cols = size[1]
    
    if cols % 2 == 1:
        xr = numpy.arange(-(cols - 1.0) / 2.0, (cols - 1.0) / 2.0, 1.0)
        xr = numpy.append(xr, (cols - 1.0) / 2.0) / float(cols - 1.0)
    else:
        xr = numpy.arange(-cols / 2.0, cols / 2.0 - 1.0, 1.0)
        xr = numpy.append(xr, cols / 2.0 - 1.0) / float(cols)

    if rows % 2 == 1:
        yr = numpy.arange(-(rows - 1.0) / 2.0, (rows - 1.0) / 2.0, 1.0)
        yr = numpy.append(yr, (rows - 1.0) / 2.0) / float(rows - 1.0)
    else:
        yr = numpy.arange(-rows / 2.0, rows / 2.0 - 1.0, 1.0) 
        yr = numpy.append(yr, rows / 2.0 - 1.0) / float(rows)
    
    x, y = numpy.meshgrid(xr, yr)
    radius = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))

    # Filter is in "unshifted" form - freq. origin at corners rather than center
    filt = numpy.power((radius / cutoff), 2 * order) + 1.0
    filt = 1.0 / filt
    filt = numpy.fft.ifftshift(filt)
    
    return filt

def highPassFilter(size, cutoff, order):
    '''
    Creates a high pass filter that can be applied to an image in the frequency domain.
    
    @param size: Size of the filter. Should be list/tuple with [num rows, num cols].
    @param cutoff: Cutoff frequency of the filter. Should be in [0, 0.5]
    @param order: Order of the filter. Higher values give sharper frequency drop offs. 
                  Should be integer >= 1.
    
    @return: High pass filter to apply to frequency-domain image
    '''
    
    return 1.0 - lowPassFilter(size, cutoff, order)

def hysThreshold(image, high, low):
    '''
    Applies hysteresis thresholding to the image. Returns binary mask where pixels in image >= high
    are on, pixels in image < low are off, and pixels < high && >= low are on iff they neighbor a
    pixel >= high.
     
    Method adapted from skimage.feature.canny.
    
    @param image: Image to threshold
    @param high: High threshold. Should be >= low.
    @param low: Low threshold. Should be <= high.
    
    @return: Hysteresis thresholded image
    '''
    
    high_mask = (image >= high)
    low_mask = (image >= low)
    
    # Segment the low-mask, then only keep low-segments that have some high_mask component in them
    strel = numpy.ones((3, 3), bool)
    labels, count = scipy.ndimage.label(low_mask, strel)
    
    if count == 0:
        return low_mask
    
    sums = scipy.ndimage.sum(high_mask, labels, numpy.arange(count, dtype=numpy.int32) + 1)
    sums = numpy.array(sums, copy = False, ndmin = 1)
    
    good_label = numpy.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    
    return output_mask

def interpImages(start, end, n, method = 'linear'):
    '''
    Interpolate values between the given start and end images. Generates a set of (equally spaced)
    intermediate images with pixel values determined by interpolating values between start and end.
    The given images should have the same dimensions.
    
    @param start: Image at one end of interpolation
    @param end: Image at other end of interpolation
    @param n: Number of intermediate, interpolated to compute
    @param method: Flag indicating intepolation method, see XXX function for description of values
    
    @return: List of n images with pixel values interpolated between values in start and end
    '''
    
    pixels = start.cols * start.rows
    
    xs = numpy.zeros((2 * pixels))
    
    xs[0 : pixels - 1] = numpy.arange(0, pixels)
    xs[pixels :] = numpy.arange(0, pixels)
    
def areaOpen(image, size, conn = None):
    '''
    Performs area opening on image. Any connected regions with fewer than size pixels are set to
    zero.
    
    @param image: Treated as binary image with 0 as background, any other value as abject
    @param size: Size threshold
    @param conn: Connectivity used to determine components. Used as in skimage.measure.label.
    
    @return: Binary image with small (<size pixels) removed
    '''
    
    labels, n = skimage.measure.label(image != 0, return_num = True, connectivity = conn)
    
    for i in range(0, n):
        idxs = numpy.where(labels == (i + 1))
        
        if idxs[0].shape[0] < size:
            labels[idxs[0], idxs[1]] = 0
        else:
            labels[idxs[0], idxs[1]] = 1
    
    return labels != 0

def fillRegions(image, size = 100, back_value = 0, fill_value = 1):
    '''
    Fills background regions (connected components back_value pixels) that are smaller than size. 
    Each pixel in the regions is replaced with fill_value.
    
    @param image: Int- or bool-valued image
    @param size: Size limit on regions to fill, regions with pixel area < size are filled
    @param back_value: Pixel value of regions to fill
    @param fill_value: Value used to fill regions 
    
    @return: Image with regions filled
    '''
    
    out = image.copy()
    labels = skimage.measure.label(image, background = -1)
    regions = skimage.measure.regionprops(labels)
    
    for r in regions:
        
        val = image[r.coords[0, 0], r.coords[0, 1]]
#         print str(r.area) + " pixels at " + str(val)
        if val == back_value and r.area < size:
            out[r.coords[:, 0], r.coords[:, 1]] = fill_value
        
    return out
    

def alpha_blend(im1, im2, alpha):
    '''
    '''
    
    blend = ((1.0 - alpha) * im2) + (alpha * im1)
    
    return blend

if __name__ == '__main__':
    '''
    Testing/debugging stuff 
    '''
    # Testing closing convolution
    input_path = '/Users/markchristopher/Documents/Data/ONH-cubes/mask_half.tif'
    input_image = skimage.io.imread(input_path) > 0
    disk = skimage.morphology.disk(100) > 0
    
    
#     image = closingByConvolution(skimage.io.imread(input_path), disk)
#     t0 = time.clock()
#     image = skimage.morphology.binary_closing(input_image, disk)
#     t = time.clock() - t0
#     print t
#     saveAsByte(image, '/Users/markchristopher/Documents/Data/ONH-cubes/def_closing.tif')
# 
#     t0 = time.clock()
#     image = mahotas.morph.close(numpy.pad(input_image, 100, 'constant'), disk)
#     t = time.clock() - t0
#     print t
#     saveAsByte(image[100:-100, 100:-100], '/Users/markchristopher/Documents/Data/ONH-cubes/new_closing.tif')

#     set = 'import mahotas; import skimage.io; import skimage.morphology; input_image = skimage.io.imread(\'/Users/markchristopher/Documents/Data/ONH-cubes/mask_half.tif\') > 0; disk = skimage.morphology.disk(10) > 0'
#     print timeit.timeit('mahotas.morph.close(input_image, disk)', setup = set, number = 10)
#     print timeit.timeit('skimage.morphology.binary_closing(input_image, disk)', setup = set, number = 10)
    
#    # Testing gaussian image
#      input_path = "/Users/mchristopher/Documents/Data/UCSD/bscans/AL0486-bscan0000.tif"
#      image = skimage.io.imread(input_path).astype(numpy.float32)
#
#     # Testing unsharp maksing
#     input_path = "/Users/mchristopher/Documents/Data/UCSD/bscans/AL0486-bscan0000.tif"
#     image = skimage.io.imread(input_path).astype(numpy.float64)
#       
#     result = unsharpMasking(image, 10.0, 2, fourier = False)
#     saveAsByte(result, '/Users/mchristopher/Documents/Data/UCSD/unsharp.tif')
#     
#     # Testing freq. filtering
#     lowpass = lowPassFilter([512, 273], 0.20, 2)
#     numpy.savetxt('/Users/mchristopher/Documents/Data/UCSD/lowpass.csv', lowpass, fmt = '%.5f', delimiter = ',')
#     saveAsByte(lowpass, '/Users/mchristopher/Documents/Data/UCSD/lowpass.tif')
#      
#     highpass = highPassFilter([512, 273], 0.20, 2)
#     numpy.savetxt('/Users/mchristopher/Documents/Data/UCSD/highpass.csv', highpass, delimiter = ',')
#     saveAsByte(highpass, '/Users/mchristopher/Documents/Data/UCSD/highpass.tif')
#      
#     boost = boostFilter([512, 273], 0.20, 2, 2.0)
#     numpy.savetxt('/Users/mchristopher/Documents/Data/UCSD/highboost.csv', boost, delimiter = ',')
#     saveAsByte(boost, '/Users/mchristopher/Documents/Data/UCSD/highboost.tif')
#      
#     boost = boostFilter([512, 273], 0.20, 2, 0.5)
#     numpy.savetxt('/Users/mchristopher/Documents/Data/UCSD/lowboost.csv', boost, delimiter = ',')
#     saveAsByte(boost, '/Users/mchristopher/Documents/Data/UCSD/lowboost.tif')
    
#     # Testing histogram truncation
#     input_path = "/Users/mchristopher/Documents/Data/UCSD/bscans/AL0486-bscan0000.tif"
#     image = skimage.io.imread(input_path).astype(numpy.float32)
#     
#     trunc = truncateHistogram(image, lo_cutoff = 20, hi_cutoff = 20)
#     saveAsByte(trunc, '/Users/mchristopher/Documents/Data/UCSD/truncated.tif')
    
#     # Testing homomorphic transform
#     input_path = "/Users/mchristopher/Documents/Data/UCSD/bscans/AL0486-bscan0000.tif"
#     image = skimage.io.imread(input_path).astype(numpy.float32)
#      
#     homo = homomorphic(image, 0, 0.2, 1, (20, 20))
# #     homo = homomorphic(image, 0, 0.2, 1)
#     saveAsByte(homo, '/Users/mchristopher/Documents/Data/UCSD/homomorphic.tif')
    
#     # Hysteresis threshold
#     input_path = "/Users/mchristopher/Documents/Data/UCSD/bscans/AL0486-bscan0000.tif"
#     image = skimage.io.imread(input_path).astype(numpy.float32)
#      
#     thresh = hysThreshold(image, 200, 100)
#     saveAsByte(thresh, '/Users/mchristopher/Documents/Data/UCSD/hysthresh.tif')
    
#     # Testing CORF
#     input_path = "/Users/mchristopher/Documents/Data/UCSD/bscans/AL0486-bscan0000.tif"
#     image = skimage.io.imread(input_path).astype(numpy.float32)
#     
#     thresh = hysThreshold(image, 200, 100)
#     saveAsByte(corf, '/Users/mchristopher/Documents/Data/UCSD/corf.tif')
    
#     # Subtract images
#     image1 = skimage.io.imread(sys.argv[1]).astype(numpy.float32)
#     image2 = skimage.io.imread(sys.argv[2]).astype(numpy.float32)
#     
#     out = image1 - image2
#     saveAsByte(out, sys.argv[3])
    
    