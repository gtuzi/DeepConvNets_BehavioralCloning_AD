# Copyright (c) 2017, Gerti Tuzi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Gerti Tuzi nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################


import scipy.ndimage as ndi
from scipy.misc import imrotate
import numpy as np

def transform_matrix_offset_center(matrix, x, y):
    ''' Taken from Keras implementation
        Incorporate to the transformation, the centering
        transformation of the image of size (x,y)
    :param matrix: transform matrix
    :param x: width of image
    :param y: height of image
    :return: transformation with respect to the center of image
    '''
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    '''
        Apply transformation matrix to an image (x).
    :param x:
    :param transform_matrix:
    :param channel_axis:
    :param fill_mode:
    :param cval:
    :return:
    '''
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def rand_horizontal_shift(X, y, shift_range = 0.0, adjust_y = False):
    '''
        Perform random shifts horizontally on the image (width-wise)
        IMPORTANT: If adjust requested:
        Apply the same fraction of horizontal shift to the continuous measurement
        (so if we shift left by 0.2, the new measurement = mesurement + 0.2*measurement)
    :param X: Image data (samples, image)
    :param y: measurement data (continuous value), shape (samples, 1)
    :param shift_range: range as a fraction ([0, 1]) of the original image
    :return:
    X_shift - shifted images
    y_shift - shifted measurement
    ordered as (X_shift, y_shift)
    '''

    # Image dimensions
    row_axis = 0
    col_axis = 1
    channel_axis = 2
    X_shift = np.zeros_like(X)
    y_shift = np.zeros_like(y)

    for i in range(0, X.shape[0]):
        p = np.random.uniform(-shift_range, shift_range) # proportion of shift
        twidth = p * X[0].shape[col_axis]
        translation_matrix = np.array([[1, 0, 0],
                                       [0, 1, twidth],
                                       [0, 0, 1]])
        h, w = X[i].shape[row_axis], X[i].shape[col_axis]
        translation_matrix = transform_matrix_offset_center(translation_matrix, h, w)
        X_shift[i] = apply_transform(x = X[i], transform_matrix = translation_matrix,
                                     channel_axis=channel_axis,
                                     fill_mode='nearest')

        if adjust_y:
            # Shift the measurement
            y_shift[i] = y[i] + y[i]*p
        else:
            y_shift[i] = y[i]

    return X_shift, y_shift

def rand_vertical_shift(X, shift_range = 0.0):
    '''
        Apply vertical random shifts.
    :param X: Data to be shifted
    :param shift_range: range as a fraction of the image height
    :return:
    Shifted data
    '''

    # Image dimensions
    row_axis = 0
    col_axis = 1
    channel_axis = 2
    X_shift = np.zeros_like(X)

    for i in range(0, X.shape[0]):
        p = np.random.uniform(-shift_range, shift_range) # proportion of shift
        theight = p * X[0].shape[row_axis]
        translation_matrix = np.array([[1, 0, theight],
                                       [0, 1, 0],
                                       [0, 0, 1]])
        h, w = X[i].shape[row_axis], X[i].shape[col_axis]
        translation_matrix = transform_matrix_offset_center(translation_matrix, h, w)
        X_shift[i] = apply_transform(x = X[i], transform_matrix = translation_matrix,
                                     channel_axis=channel_axis,
                                     fill_mode='nearest')

    return X_shift

def rand_shear_shift(X, shear_range = 0.0):
    '''
        Apply random shear (rotation in degrees) transformation to list of images
    :param X: Image list of shape ([sample], [width], [height], [chanels])
    :param shear_range: range of sheer in degrees
    :return: shear rotated image list of same size as input
    '''

    row_axis = 0
    col_axis = 1
    channel_axis = 2

    range_rad = np.pi / 180 * shear_range

    X_shear = np.zeros_like(X)

    for i in range(0, X.shape[0]):
        shear = np.random.uniform(-range_rad, range_rad)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        h, w = X[i].shape[row_axis], X[i].shape[col_axis]
        translation_matrix = transform_matrix_offset_center(shear_matrix, h, w)
        X_shear[i] = apply_transform(x=X[i], transform_matrix=translation_matrix,
                                     channel_axis=channel_axis,
                                     fill_mode='nearest')
    return X_shear

def rand_rotations(X, rotation_range):
    # Iterate through the list of images and rotate
    # X : numpy array with shape [samples][w][h][channels]
    # rotation_range: angle (in degrees) range from which to uniformly randomly sample

    X_rot = np.zeros_like(X)
    for i in range(0, X.shape[0]):
        theta = np.random.uniform(-rotation_range, rotation_range)
        X_rot[i] = imrotate(X[i], theta)

    return X_rot