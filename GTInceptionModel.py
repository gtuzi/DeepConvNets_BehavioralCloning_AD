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


import keras
from keras.layers import Dense, Convolution2D, MaxPooling2D, \
    Dropout, Lambda, Cropping2D, Concatenate, Input, Flatten, Merge
from keras.models import Model
from keras import backend as K
import tensorflow as tf

def GTInceptionModel(input_shape, drop_prob):
    '''
        GT-Incetption leverages the inception module
    :param input_shape: shape of the image
    :param drop_prob: drop probability
    :return: GTInception (Keras) model
    '''

    mult = 10 # Filter multiplier. A brute way to build capacity the "width" way

    x_input = Input(shape=input_shape)

    # Pre-process the input: Crop the input
    top_crop = 50
    bottom_crop = 20
    x = Cropping2D(cropping=((top_crop, bottom_crop), (0, 0)))(x_input)

    # Center pixel values
    x = Lambda(lambda xi: xi / 255.0 - 0.5)(x)

    # ---------------- Layer 1 ---------------------
    # Layer 1 is a regular old 2D convolution layer
    # Using the strides of the convolution for a
    # quick drop of image size
    # 160, 320, 3 --> 78, 158, 6 * mult
    XC1 = Convolution2D(filters=6*mult,  # the dimensionality of the output space
                            kernel_size=(5, 5),  # width and height of the 2D convolution window
                            strides=(2, 2),  # strides of the convolution along the width and height
                            padding='valid',
                            data_format='channels_last',  # The ordering of the dimensions in the inputs.
                                                          # `channels_last` corresponds to inputs with shape
                                                          # `(batch, width, height, channels)`
                            activation='relu',
                            kernel_initializer='he_normal')(x)



    # ------------ Inception Layer 2 --------------- #
    # 78, 158, 6 * mult to 39 * 79 * 6*mult
    i2k1 = 3 * mult
    i2k3 = 1 * mult
    i2k5 = 1 * mult
    i2kp = 1 * mult
    XInc2 = InceptionModule(x=XC1, k1=i2k1, k3=i2k3, k5=i2k5, kp=i2kp)
    XInc2 = MaxPooling2D(pool_size=(2,2), padding='same')(XInc2)


    # ------------ Inception Layer 3 --------------- #
    # 39 * 79 * 6*mult to 19 * 39 * 10*mult
    i3k1 = 3 * mult
    i3k3 = 3 * mult
    i3k5 = 2 * mult
    i3kp = 2 * mult
    XInc3 = InceptionModule(x=XInc2, k1=i3k1, k3=i3k3, k5=i3k5, kp=i3kp)
    XInc3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(XInc3)



    # ------------ Inception Layer 4 --------------- #
    # 19 * 39 * 10*mult to 19 * 39 * 16*mult
    i4k1 = 7 * mult
    i4k3 = 5 * mult
    i4k5 = 2 * mult
    i4kp = 2 * mult
    XInc4 = InceptionModule(x=XInc3, k1=i4k1, k3=i4k3, k5=i4k5, kp=i4kp)


    # ------------ Inception Layer 5 --------------- #
    # 19 * 39 * 16*mult to 9 * 19 * 16*mult
    i5k1 = 5 * mult
    i5k3 = 6 * mult
    i5k5 = 3 * mult
    i5kp = 2 * mult
    XInc5 = InceptionModule(x=XInc4, k1=i5k1, k3=i5k3, k5=i5k5, kp=i5kp)
    XInc5 = MaxPooling2D(pool_size=(2, 2), padding='valid')(XInc5)


    # ------------ Inception Layer 6 --------------- #
    # 9 * 19 * 16*mult to 9 * 19 * 32*mult
    i6k1 = 10 * mult
    i6k3 = 14 * mult
    i6k5 = 5 * mult
    i6kp = 3 * mult
    XInc6 = InceptionModule(x=XInc5, k1=i6k1, k3=i6k3, k5=i6k5, kp=i6kp)


    # ------------ Inception Layer 7 --------------- #
    # 9 * 19 * 32*mult to 4 * 9 * 64*mult
    i7k1 = 17 * mult
    i7k3 = 32 * mult
    i7k5 = 10 * mult
    i7kp = 5 * mult
    XInc7 = InceptionModule(x=XInc6, k1=i7k1, k3=i7k3, k5=i7k5, kp=i7kp)

    # ------------ Flatten Max-Pool 8 --------------- #
    # 4 * 9 * 64*mult to 64*mult
    shp = keras.backend.int_shape(XInc7)[1:3]
    MP8 = MaxPooling2D(pool_size=shp, padding='valid')(XInc7)
    MP8 = Flatten()(MP8)
    MP8 = Dropout(rate=drop_prob)(MP8)

    # ------------ Layer 9: Fully Connected Layer --------------- #
    # 64*mult to 1024
    FC9 = Dense(units=1024, # dimensionality of the output space
                activation='relu',
                kernel_initializer='normal')(MP8)

    # Regression output
    logit = Dense(units=1, kernel_initializer='normal')(FC9)


    # Tie things up in the model
    model = Model(inputs=x_input, outputs=logit)





def InceptionModule(x, k1, k3, k5, kp):
    '''
        Inception module (a naiive implementation)
    :param x: input tensor from previous layer
    :param k1: depth of the 1x1 convolution
    :param k3: depth of the 3x3 convolution
    :param k5: depth of the 5x5 convolution
    :param kp: depth of the max-pooling layer
    :return: Inception output tensor
    '''

    # 1x1 convolution
    C1 = Convolution2D(filters=k1,  # the dimensionality of the output space
                        kernel_size=(1, 1),  # width and height of the 2D convolution window
                        strides=(1, 1),  # strides of the convolution along the width and height
                        padding='same',
                        data_format='channels_last',  # The ordering of the dimensions in the inputs.
                                                      # `channels_last` corresponds to inputs with shape
                                                      # `(batch, width, height, channels)`
                        activation='relu',
                        kernel_initializer='he_normal')(x)


    # 3x3 convolution
    C3 = Convolution2D(filters=k3,  # the dimensionality of the output space
                        kernel_size=(3, 3),  # width and height of the 2D convolution window
                        strides=(1, 1),  # strides of the convolution along the width and height
                        padding='same',
                        data_format='channels_last',  # The ordering of the dimensions in the inputs.
                                                      # `channels_last` corresponds to inputs with shape
                                                      # `(batch, width, height, channels)`
                        activation='relu',
                        kernel_initializer='he_normal')(C1)

    # 5 x 5 convolution
    C5 = Convolution2D(filters=k5,  # the dimensionality of the output space
                       kernel_size=(5, 5),  # width and height of the 2D convolution window
                       strides=(1, 1),  # strides of the convolution along the width and height
                       padding='same',
                       data_format='channels_last',  # The ordering of the dimensions in the inputs.
                       # `channels_last` corresponds to inputs with shape
                       # `(batch, width, height, channels)`
                       activation='relu',
                       kernel_initializer='he_normal')(C1)

    # 3x3 max-pooling followed by a 1x1 2D Convolution
    MP2 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)

    CMP = Convolution2D(filters=kp,         # the dimensionality of the output space
                       kernel_size=(1,1),   # width and height of the 2D convolution window
                       strides=(1,1),       # strides of the convolution along the width and height
                       padding='same',
                       data_format='channels_last',  # The ordering of the dimensions in the inputs.
                                                     # `channels_last` corresponds to inputs with shape
                                                     # `(batch, width, height, channels)`
                       activation='relu',
                       kernel_initializer='he_normal')(MP2)


    # Concatenate the convolutions
    spatial_shape = keras.backend.int_shape(C1)[1:3]
    concat_depth = (k1 + k3 + k5 + kp,)
    output_shape = spatial_shape + concat_depth

    # IOut = tf.concat(concat_dim=3, values=[C1, C3, C5, CMP])
    # IOut = keras.layers.concatenate(inputs=[C1, C3, C5, CMP], axis=1)
    IOut = Lambda(lambda xi:xi)([C1, C3, C5, CMP])
    return IOut



