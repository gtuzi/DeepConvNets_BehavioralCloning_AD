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



from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, \
    Dropout, SpatialDropout2D, GlobalAveragePooling2D, Lambda, Cropping2D



def GTRegressionModel(input_shape, drop_prob):
    '''
        A general deep 2D Conv-Net for regression
    :param input_shape:
    :param drop_prob:
    :return:
    '''
    model = Sequential()

    # Pre-process the input
    top_crop = 50
    bottom_crop = 20
    model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0, 0)), input_shape=input_shape))

    # Center
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    # ---------------- Layer 1 ---------------------

    # 160, 320, 3 --> 78, 158, 12
    model.add(Convolution2D(filters=12,                  # the dimensionality of the output space
                            kernel_size=(5, 5),          # width and height of the 2D convolution window
                            strides=(2,2),               # strides of the convolution along the width and height
                            padding='valid',
                            data_format='channels_last', # The ordering of the dimensions in the inputs.
                                                         #`channels_last` corresponds to inputs with shape
                                                         # `(batch, width, height, channels)`
                            activation='relu',
                            kernel_initializer='he_normal',
                            name='Conv1'))


    # Pool to aggressively drop the size
    # 78, 158, 12 --> 39, 79, 12
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid', name='MaxPool1'))

    # ---------------- Layer 2 ---------------------
    # Size drops using convolution
    #
    # 39, 79, 12 --> 18, 38, 48
    model.add(Convolution2D(filters=48,
                            kernel_size=(4, 4),
                            strides=(2,2),
                            padding='valid',
                            data_format='channels_last',
                            activation='relu',
                            kernel_initializer='he_normal',
                            name='Conv2'))

    model.add(SpatialDropout2D(rate=drop_prob))

    # ---------------- Layer 3 ---------------------
    # 18, 38, 48 --> 8, 18, 64
    model.add(Convolution2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(2,2),
                            padding='valid',
                            data_format='channels_last',
                            activation='relu',
                            kernel_initializer='he_normal',
                            name='Conv3'))

    model.add(SpatialDropout2D(rate=drop_prob))

    # ---------------- Layer 4 ---------------------
    # 8, 18, 64 --> 64
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(rate=drop_prob))

    # Fully connected (128)
    model.add(Dense(units=128, # dimensionality of the output space
                    kernel_initializer='he_normal'))

    # ---------------- Output Layer -----------------
    # Outputting only the logits
    model.add(Dense(units=1, kernel_initializer='normal'))

    return model