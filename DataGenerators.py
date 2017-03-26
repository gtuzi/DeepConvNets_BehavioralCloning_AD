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



from sklearn.utils import shuffle
import cv2
import numpy as np
from DataAgumentation import rand_horizontal_shift, rand_vertical_shift, rand_rotations, rand_shear_shift
import math

def get_image(filename):
    filename = filename.split('/')[-1]
    filename = 'data/IMG/' + filename
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def process_log_line_3cam_angle(line):
    center_cam_token = 0
    left_cam_token = 1
    right_cam_token = 2

    images = []
    measurements = []

    # Get images
    center_filename = line[center_cam_token]
    center_image = get_image(center_filename)
    left_filename = line[left_cam_token]
    left_image = get_image(left_filename)
    right_filename = line[right_cam_token]
    right_image = get_image(right_filename)

    # Chain the images
    images.extend([center_image, left_image, right_image])

    correction = 0.2
    center_measurement = float(line[3])
    left_measurement = center_measurement + correction
    right_measurement = center_measurement - correction

    # Chain the steering measurements
    measurements.extend([center_measurement, left_measurement, right_measurement])

    # Flip image horizontally and measurement
    center_image_flipped = np.fliplr(center_image)
    left_image_flipped = np.fliplr(left_image)
    right_image_flipped = np.fliplr(right_image)
    images.extend([center_image_flipped, left_image_flipped, right_image_flipped])

    center_measurement_flipped = -center_measurement
    left_measurement_flipped = -left_measurement
    right_measurement_flipped = -right_measurement
    measurements.extend([center_measurement_flipped, left_measurement_flipped, right_measurement_flipped])

    return images, measurements


def process_log_line_center_cam_angle(line):
    center_cam_token = 0
    angle_token = 3
    images = []
    measurements = []

    # Get images
    center_filename = line[center_cam_token]
    center_image = get_image(center_filename)
    # Chain the images
    images.append(center_image)

    center_measurement = float(line[angle_token])
    # Chain the steering measurements
    measurements.append(center_measurement)

    # Flip image horizontally and measurement
    center_image_flipped = np.fliplr(center_image)
    images.append(center_image_flipped)
    center_measurement_flipped = -center_measurement
    measurements.append(center_measurement_flipped)

    return images, measurements


def process_log_line_3cam_vel(line):
    '''
    Process all cams and velocity (corrected for shifted)
    :param line:
    :return:
    '''
    vel_token = 6
    center_cam_token = 0
    left_cam_token = 1
    right_cam_token = 2

    images = []
    measurements = []

    # Get images & measurements
    center_filename = line[center_cam_token]
    center_image = get_image(center_filename)
    left_filename = line[left_cam_token]
    left_image = get_image(left_filename)
    right_filename = line[right_cam_token]
    right_image = get_image(right_filename)
    velocity_measurement = float(line[vel_token])

    # Chain the images & measurements
    images.extend([center_image, left_image, right_image])

    # Correct the velocity if we've gone out of bounds
    # but slow down, we've gone out of bounds
    correction = 5
    left_measurement =  np.max([3, velocity_measurement - correction])
    right_measurement = np.max([3, velocity_measurement - correction])
    measurements.extend([velocity_measurement, left_measurement, right_measurement])

    # Flip image horizontally and measurement
    center_image_flipped = np.fliplr(center_image)
    left_image_flipped = np.fliplr(left_image)
    right_image_flipped = np.fliplr(right_image)

    # For the flipped images, still keep the same velocity
    images.extend([center_image_flipped, left_image_flipped, right_image_flipped])
    measurements.extend([velocity_measurement, left_measurement, right_measurement])

    return images, measurements


def process_log_line_center_cam_vel(line):
    '''
        Process center cam and veolocity
    :param line:
    :return:
    '''

    vel_token = 6
    center_cam_token = 0

    images = []
    measurements = []

    # Get images & measurements
    center_filename = line[center_cam_token]
    center_image = get_image(center_filename)
    velocity_measurement = float(line[vel_token])

    images.append(center_image)
    measurements.append(velocity_measurement)

    return images, measurements


def train_generator_3(loglines, sixth_of_batch_size=32,
                    horizontal_augment=False,
                    horizontal_shift_range=0.2,
                    vertical_augment=False,
                    vertical_shift_range=0.2,
                    rotation_augment=False,
                    rotation_range=10.,
                    shear_augment=False,
                    shear_range =10.,
                    do_shuffle=True,
                    mode = 'angle'):

    '''
    Each line in the log file contains references to 3 camera outputs.
    Each sample in the batch will contain 3 camera images.
    Each image will be flipped horizontally
    Therefore, the number of images that will be returned is 3 * 2 = 6 times the
    sixth_of_batch_size.
    :param loglines: log-file lines
    :param sixth_of_batch_size: batch size is one sample on the log fie
    :param horizontal_augment: apply horizontal augmentation. This will affect the angle (y) measurement
                               if in 'angle' mode
    :param horizontal_shift_range: uniform random horizontal shift range as a portion of the horizontal size.
                                   Uniform range in [-range, +range]
                                   If in 'angle' mode:
                                   This same portion will be applied to the angle measurement (y) as:
                                   <new angle> = <old angle> + (horizontal_shift_range)*<old angle>
    :param vertical_augment: apply vertical augmentation
    :param vertical_shift_range: vertical shift range as a portion of the vertical size
    :param rotation_augment: apply rotation augmentation
    :param rotation_range: angle in degrees range [-range, +range] for which to
                           uniformly apply rotation to image
    :param shear_augment: apply shear rotations
    :param shear_range: ngle in degrees range [-range, +range] for which to
                           uniformly apply shear rotation to image
    :param mode: 'angle' (default) or 'velocity'
    :return: (X, y) batch
    '''

    num_samples = len(loglines)
    while 1: # Loop forever so the generator never terminates
        shuffle(loglines)
        for offset in range(0, num_samples, sixth_of_batch_size):
            batch_log_lines = loglines[offset:offset+sixth_of_batch_size]

            images = []
            measurements = []
            for line in batch_log_lines:
                if mode == 'velocity':
                    imgs, meas = process_log_line_3cam_vel(line)
                else:
                    imgs, meas = process_log_line_3cam_angle(line)

                images.extend(imgs)
                measurements.extend(meas)

            X = np.array(images)
            y = np.array(measurements)

            if horizontal_augment:
                if mode == 'velocity':
                    [X, _] = rand_horizontal_shift(X, y, shift_range=horizontal_shift_range, adjust_y=False)
                else:
                    # Horizontal shift adjusts the angle measure !!!
                    [X, y] = rand_horizontal_shift(X, y, shift_range=horizontal_shift_range, adjust_y=True)
            if vertical_augment:
                X = rand_vertical_shift(X=X, shift_range=vertical_shift_range)
            if rotation_augment:
                X = rand_rotations(X=X, rotation_range= rotation_range)
            if shear_augment:
                X = rand_shear_shift(X=X, shear_range=shear_range)


            if do_shuffle:
                yield shuffle(X, y)
            else:
                yield (X,y)


def train_generator_center(loglines, half_batch_size=32,
                    horizontal_augment=False,
                    horizontal_shift_range=0.2,
                    vertical_augment=False,
                    vertical_shift_range=0.2,
                    rotation_augment=False,
                    rotation_range=10.,
                    shear_augment=False,
                    shear_range =10.,
                    do_shuffle=True,
                    mode='angle'):

    '''
    Each line in the log file contains references to 3 camera outputs.
    Each sample in the batch will contain center camera image.
    Center image will be flipped, yielding a doubling in samples
    Each image will be flipped horizontally
    :param loglines: log-file lines
    :param batch_size: batch size is one sample on the log fie
    :param horizontal_augment: apply horizontal augmentation.
    :param horizontal_shift_range: uniform random horizontal shift range as a portion of the horizontal size.
                                   In [-range, +range]
    :param vertical_augment: apply vertical augmentation
    :param vertical_shift_range: vertical shift range as a portion of the vertical size
    :param rotation_augment: apply rotation augmentation
    :param rotation_range: angle in degrees range [-range, +range] for which to
                           uniformly apply rotation to image
    :param shear_augment: apply shear rotations
    :param shear_range: ngle in degrees range [-range, +range] for which to
                           uniformly apply shear rotation to image
    :param mode: 'angle' (default) or 'velocity'
    :return: (X, y) batch
    '''

    num_samples = len(loglines)
    while 1: # Loop forever so the generator never terminates
        shuffle(loglines)
        for offset in range(0, num_samples, half_batch_size):
            batch_log_lines = loglines[offset:offset+half_batch_size]

            images = []
            measurements = []
            for line in batch_log_lines:
                if mode == 'velocity':
                    imgs, meas = process_log_line_center_cam_vel(line)
                else:
                    imgs, meas = process_log_line_center_cam_angle(line)

                images.extend(imgs)
                measurements.extend(meas)

            X = np.array(images)
            y = np.array(measurements)

            if horizontal_augment:
                [X, _] = rand_horizontal_shift(X, y, shift_range=horizontal_shift_range, adjust_y=False)
            if vertical_augment:
                X = rand_vertical_shift(X=X, shift_range=vertical_shift_range)
            if rotation_augment:
                X = rand_rotations(X=X, rotation_range= rotation_range)
            if shear_augment:
                X = rand_shear_shift(X=X, shear_range=shear_range)


            if do_shuffle:
                yield shuffle(X, y)
            else:
                yield (X,y)


def generator(loglines, batch_size = 32, mode ='angle'):

    '''
    Obtain only the center camera image and the center angle measurement
    :param loglines:
    :param batch_size:
    :param mode: angle (default) or velocity
    :return:
    '''
    center_cam_token = 0
    num_samples = len(loglines)
    angle_token = 3
    velo_token = 6
    while 1:  # Loop forever so the generator never terminates
        shuffle(loglines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = loglines[offset:offset + batch_size]

            images = []
            measurement = []
            for line in batch_lines:
                center_filename = line[center_cam_token]
                center_image = get_image(center_filename)
                if mode == 'velocity':
                    meas = float(line[velo_token])
                else:
                    meas = float(line[angle_token])

                images.append(center_image)
                measurement.append(meas)

            # trim image to only see section with road
            X = np.array(images)
            y = np.array(measurement)
            yield shuffle(X, y)