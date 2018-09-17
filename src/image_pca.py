'''
Created on Oct 14, 2016

@author: mchristopher
'''

import os
import sys
import os.path

import numpy
import numpy.ma
import skimage.io
import sklearn.decomposition

import argparse

import imtools

class ImagePCA:
    '''
    A principal component analysis (PCA) reduction of a set of images.
    
    Images are vectorized and standardized (zero mean, unit variance) prior to application of PCA.
    All images should have the same dimensions. Optionally, a binary mask can be applied to the 
    input images to select regions to include/ignore during PCA 
    '''
    # Data extracted from images, each vectorized image store as a row
    # List on which PCA is applied
#     image_list = None
#     
#     # Binary mask applied to input images prior to PCA
#     mask = None
#     
#     # Mean used to standardize image prior to PCA
#     mean = None
#     
#     # Standard deviation used to standardize image prior to PCA
#     std = None
#     
#     # PCA object used to perform decomposition
#     pca = None
    
    def __init__(self, file_list, mean = numpy.empty(0), std = numpy.empty(0), mask = numpy.empty(0)):
        '''
        Create instance to perform PCA on the given input list.
        
        If mean/std are provided, they are used to standardize data prior to PCA. Otherwise, they
        are computed from the data.
        
        @param file_list: 
        @param mean: 
        @param std: 
        @param mask: 
         
        '''
        
        first = True
        
        # Read each image
        for i in range(len(file_list)):
            image = skimage.io.imread(file_list[i])
            
            # Get mask (or use whole image)
            if first:
                if mask.size == 0:
                    self.mask = numpy.zeros(image.shape, numpy.uint8)
                else:
                    self.mask = numpy.logical_not(mask.astype(numpy.bool8))
                    
            
            masked = numpy.ma.array(image, mask = self.mask, dtype = numpy.float32)
            row = masked.compressed()
            
            if first:
                self.data = numpy.zeros((len(file_list), row.size), dtype = numpy.float32)
                first = False
                
            self.data[i, :] = row
        
        # Compute mean/std (or use provided values) to standardize
        if mean.size == 0:
            self.mean = numpy.mean(self.data, axis = 0)
            self.std = numpy.std(self.data, axis = 0)
        else:
            self.mean = mean
            self.std = std
        
        
        for i in range(self.data.shape[0]):
            self.data[i, :] = self.data[i, :] - self.mean
            self.data[i, :] = self.data[i, :] / self.std
        
        # Ready PCA object
        self.pca = sklearn.decomposition.PCA()
        self.transformed = None
        
    
    def fitPCA(self):
        '''
        Fit PCA model to the input images.
        '''
        self.pca.fit(self.data)
        self.transformed = self.pca.transform(self.data)
    
    def getPCImage(self, idx):
        '''
        Returns image representation of the PC of the given index.
        
        @param idx: Index of PC to return
        '''
        
        pc = numpy.zeros(self.mask.shape)
        pc[numpy.where(numpy.logical_not(self.mask))] = self.pca.components_[idx, :]
        
        return pc

    def getMeanImage(self):
        '''
        Returns mean input image (applies any assigned mask).
        
        @return: Mean input image
        '''
        
        m_image = numpy.zeros(self.mask.shape)
        m_image[numpy.where(numpy.logical_not(self.mask))] = self.mean
        
        return m_image
    
    def getStdImage(self):
        '''
        Returns st. dev. input image (applies any assigned mask).
        
        @return: St. dev. input image
        '''
        
        s_image = numpy.zeros(self.mask.shape)
        s_image[numpy.where(numpy.logical_not(self.mask))] = self.std
        
        return s_image
    
    def getCOVImage(self):
        '''
        Returns coefficient of variation input image (applies any assigned mask).
        
        COV = st. dev. / mean
        
        @return: COV input image
        '''
        
        c_image = numpy.zeros(self.mask.shape)
        c_image[numpy.where(numpy.logical_not(self.mask))] = numpy.divide(self.std, self.mean)
        
        return c_image
    
    def getEigenValue(self, idx, normalized = False):
        '''
        Returns eigen value corresponding to the PC given by idx.
        
        @param idx: The PC index
        @param normalized: If True, returns normalized value (divided by sum of eigen values).
        
        @return: Eigen value
        '''
        
        if normalized:
            return self.pca.explained_variance_ratio_[idx]
        else:
            return self.pca.explained_variance_[idx]
        
    
    def getNumPCs(self):
        '''
        Returns total number of PCs computed.
        
        @return: Number of components
        '''
        return self.pca.components_.shape[0]
    
        
if __name__ == '__main__':
    '''
    
    Apply PCA to a given set of images. Produces PC images and data projected
    into the PCA space as output.
    
    '''
    
    clargs = argparse.ArgumentParser()
    
    # Required arguments
    clargs.add_argument(
        'list_path',
        help = 'Path to text file listing set of input images.')
     
    # Optional arguments
    clargs.add_argument(
        '-o', '--out_dir',
        default = './',
        help = "Path to store output.")
    clargs.add_argument(
        '-m', '--mask_path',
        default = None,
        help = "Binary image used to mask regions to ignore during PCA.")
    clargs.add_argument(
        '-s', '--stand_path',
        default = None,
        help = "Mean / std. deviation used to standardize data. Should be *.npz file containing " + 
                "two numpy arrays, named \"mean\" and \"std\".")
    clargs.add_argument(
        '-l', '--leaveout',
        default = None,
        help = "Path to file containing binary vector indicating input files to exclude during model building. " + 
                "The images are projected onto the computed PCA basis and output in 'excluded-projected.csv'")
    
    args = clargs.parse_args()
    
    # Create output destination
    if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
    
    # Get input list
    f = open(args.list_path, 'r') 
    image_list = f.readlines()
    image_list = [i.strip() for i in image_list]
    f.close()
    
    # Get optional mask 
    if args.mask_path:
        mask = skimage.io.imread(args.mask_path)
    else:
        mask = numpy.empty(0)
    
    if args.leaveout:
        in_list = list()
        out_list = list()
        exclude = numpy.loadtxt(args.leaveout, dtype = numpy.uint8, delimiter = ',')
        
        for i in range(0, len(image_list)):
            if not exclude[i] == 0:
                out_list.append(image_list[i])
            else:
                in_list.append(image_list[i])
        
        image_list = in_list
        
    
    # Build PCA model
    pca = ImagePCA(image_list, mask = mask)
    pca.fitPCA()
    
    # Save PCA model data
    if not args.stand_path:
        path = '%s/meanstd.npz' % (args.out_dir)
        numpy.savez(path, mean = pca.mean, std = pca.std)
    
    path = '%s/mean.tif' % (args.out_dir)
    imtools.saveAsByte(pca.getMeanImage(), path)
    
    path = '%s/stdev.tif' % (args.out_dir)
    imtools.saveAsByte(pca.getStdImage(), path)
    
    path = '%s/cov.tif' % (args.out_dir)
    imtools.saveAsByte(pca.getCOVImage(), path)
    
    header = ""
    for i in range(pca.getNumPCs()):
        header += "pc" + str(i + 1) + ","
    
    path = '%s/pcarep.csv' % (args.out_dir)
    numpy.savetxt(path, pca.transformed, fmt = '%.8g', delimiter = ",", header = header[:-1], comments = "")
    
    path = '%s/data.csv' % (args.out_dir)
    numpy.savetxt(path, pca.data, fmt = '%.8g', delimiter = ",", comments = "")
    
    path = '%s/pcs.csv' % (args.out_dir)
    numpy.savetxt(path, pca.pca.components_, fmt = '%.8g', delimiter = ",")
    
    path = '%s/eigs.csv' % (args.out_dir)
    eigs = numpy.zeros((pca.pca.explained_variance_.shape[0], 2))
    eigs[:, 0] = pca.pca.explained_variance_
    eigs[:, 1] = pca.pca.explained_variance_ratio_
    numpy.savetxt(path, eigs, fmt = '%.8g', delimiter = ",")
    
    # Save PC images
    for i in range(pca.getNumPCs()):
        path = '%s/pc%05d.tif' % (args.out_dir, i + 1)
        
        r = 14
        c = 7
        
        cur_image = pca.getPCImage(i)
        cur_image = cur_image[c : cur_image.shape[0] - c, r : cur_image.shape[1] - r]
        cur_image = numpy.abs(cur_image)
        
        imtools.saveAsByte(cur_image, path)
        print str(pca.getEigenValue(i, True)) + " (" + str(pca.getEigenValue(i, False)) + ")"
        
    
    if args.leaveout:
#     if True:
#         out_list = image_list
        out_pca = ImagePCA(out_list, mask = mask, mean = pca.mean, std = pca.std)
        projected = pca.pca.components_.dot(numpy.transpose(out_pca.data))
        projected = numpy.transpose(projected)
        
        path = '%s/excluded-projected.csv' % (args.out_dir)
        numpy.savetxt(path, projected, fmt = '%.8g', delimiter = ",", header = header[:-1])
        
    