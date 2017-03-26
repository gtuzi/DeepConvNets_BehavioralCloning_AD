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


import csv
import cv2
import numpy as np

# -------------------------
# Load data from the csv file
loglines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        loglines.append(line)


from sklearn.model_selection import train_test_split
validation_portion = 0.3
# Log lines are our references to the samples
train_loglines, test_loglines = train_test_split(loglines, test_size=0.333, random_state=0)
train_loglines, validation_loglines = train_test_split(train_loglines, test_size=validation_portion, random_state=0)

num_cams = 3
# 3 cameras * (1 original + 1 horizontal flip)
n_train = len(train_loglines) * 2
n_valid = len(validation_loglines)
n_test = len(test_loglines)

print('Num Train Samples: ' + str(n_train))
print('Num Valid Samples: ' + str(n_valid))
print('Num Test Samples: ' + str(n_test))

# -------------------------
# Using images as the input and steering angle as the output
# create a regression model that predicts the continuous value
# of the output value


# images = []
# measurements = []
#
# for line in lines:
#     imgs, meas = process_log_line(line)
#     images.extend(imgs)
#     measurements.extend(meas)


# for line in lines:
#     sourcepath = line[center_cam_token]
#     filename = sourcepath.split('/')[-1]
#     current_path = 'data/IMG/' + filename
#     image = cv2.imread(current_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     images.append(image)
#     measurement = float(line[3])
#     measurements.append(measurement)
#
#     # Flip image horizontally and measurement
#     image_flipped = np.fliplr(image)
#     images.append(image_flipped)
#     measurement_flipped = -measurement
#     measurements.append(measurement_flipped)



# ------------------------------------------------
# Keras works with numpy arrays
# n_samps = len(measurements)
# X_orig = np.array(images)
# y_orig = np.array(measurements)

# ------------------------------------------------
# Double the measurements with random horizontal shifts
# from DataAgumentation import rand_horizontal_shift
#
# [X_shifted, y_shifted] = rand_horizontal_shift(X_orig, y_orig, shift_range=0.2)
#
# X = np.zeros(shape=(2*n_samps,) + X_orig.shape[1:], dtype=X_orig.dtype)
# y = np.zeros(shape=(2*n_samps, 1), dtype=y_orig.dtype)

# ------------------------------------
# Keras data Augmentation
# from keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split
# from keras import backend as K
# K.set_image_dim_ordering('tf')
# validation_portion = 0.3
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_portion, random_state=0)
#
# traindatagen = ImageDataGenerator(rotation_range=20.)
# traindatagen.fit(X_train)
#
# validdatagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.3, height_shift_range=0.3)
# validdatagen.fit(X_valid)
#
# import matplotlib.pyplot as plt
# for X_batch, y_batch in traindatagen.flow(X_train, y_train, batch_size=2, seed=0):
#     # create a grid of
#     for i in range(0, 2):
#         plt.subplot(220 + 1 + i)
#         img = X_batch[i].squeeze().astype(np.uint16)
#         plt.imshow(img)
#         #plt.imshow(cv2.cvtColor(img.astype(np.int32), cv2.COLOR_BGR2RGB))
#
#     # show the plot
#     plt.show()
#     break
#
# ------------------------------------
# Train the model (very simple model)

from GTRegressionModel import GTRegressionModel
from GTInceptionModel import GTInceptionModel
import math, os
from keras.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler
from DataGenerators import train_generator_center, train_generator_3, generator

model_dir = 'model'
best_model = model_dir + '/model.h5'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Training parameters
Initial_LR = 0.001
Train_batch_size = 36  # multiples of 2
Valid_batch_size = 32
Test_batch_size = 32
Dropout_Rate = 0.8
Max_No_Epochs = 10

Num_Batches_Per_Train_Epoch = int(n_train / Train_batch_size)
if (n_train % Train_batch_size) > 0:
    Num_Batches_Per_Train_Epoch += 1

Num_Batches_Per_Valid_Epoch = int(n_valid / Valid_batch_size)
if (n_valid % Valid_batch_size) > 0:
    Num_Batches_Per_Valid_Epoch += 1

Num_Batches_Per_Test_Epoch = int(n_test / Test_batch_size)
if (n_test % Test_batch_size) > 0:
    Num_Batches_Per_Test_Epoch += 1


train_gen_angle = train_generator_center(train_loglines, half_batch_size=int(Train_batch_size/2.),
                            horizontal_augment=False, horizontal_shift_range=0.1,
                            vertical_augment=False, vertical_shift_range=0.1,
                            rotation_augment=False, rotation_range=5.,
                            shear_augment=False, shear_range=5.,
                            do_shuffle=True)
test_gen_angle = generator(loglines=test_loglines, batch_size=Test_batch_size)
valid_gen_angle = generator(loglines=validation_loglines, batch_size=Valid_batch_size)


train_gen_velocity = train_generator_3(train_loglines, sixth_of_batch_size=int(Train_batch_size/2.),
                            horizontal_augment=True, horizontal_shift_range=0.1,
                            vertical_augment=False, vertical_shift_range=0.1,
                            rotation_augment=False, rotation_range=5.,
                            shear_augment=False, shear_range=5.,
                            do_shuffle=True, mode='velocity')

test_gen_velocity = generator(loglines=test_loglines, batch_size=Test_batch_size, mode='velocity')
valid_gen_velocity = generator(loglines=validation_loglines, batch_size=Valid_batch_size, mode='velocity')


# learning rate schedule
def step_decay(epoch):
    # Number of batches before learning rate drops by a 'drop_factor'
    batch_count_drop_step = 8000.0
    drop_factor = 0.1

    batch_count = float(epoch*Num_Batches_Per_Train_Epoch)
    pow_factor = (math.pow(drop_factor, math.floor(batch_count/(batch_count_drop_step))))
    lrate = Initial_LR*pow_factor

    print("Epoch: {0} - Batch_count: {1} - Pow Fact: {2:0.3f} - lrate: {3:0.6f}".
          format(epoch, batch_count, pow_factor, lrate))
    return lrate


# learning schedule callback
lrate = LearningRateScheduler(step_decay)


# The best model here is determined by the monitor parameter.
# It can be the best based on training values, or validataion (prefixed by val_)
best_val_mse_checkpoint = ModelCheckpoint(best_model,
                             monitor= 'val_loss', # This determines what gets saved (val_acc: validation accuracy)
                             verbose=1,
                             save_best_only=True,
                             save_weights_only= False, # If false the whole model is saved
                             mode='auto')


earlystopping = EarlyStopping(monitor='val_loss',
                              #  minimum change in the monitored quantity to qualify as an improvement,
                              # i.e. an absolute change of less than min_delta, will count as no improvement.
                              min_delta=0.0,
                              # number of epochs with no improvement after
                              # which training will be stopped
                              patience=30,
                              verbose=1,
                              mode='auto')

callbacks_list = [best_val_mse_checkpoint, earlystopping, lrate]

model = GTRegressionModel(input_shape=(160, 320, 3), drop_prob=Dropout_Rate)

# Minimizing the error between the logits and the
# continuous value of the labels (steering wheel angle)
# we get a regression fit (as opposed to classification)
# Using the loss function MSE - the same loss function
# used in regular regression
model.compile(loss='mse', optimizer='adam')


print("************** Parameters **********************")
print (("- Train Batch_Size : {0}\n- Max_No_Epochs: {1}\n" +
        "- Init_LR: {2}\n- Num_Batches_Per_Epoch: {3}\n" +
        "- Dropout_Rate: {4:0.3f}").format(
    Train_batch_size, Max_No_Epochs,
    Initial_LR,Num_Batches_Per_Train_Epoch,Dropout_Rate))
print("***************************************************")


# Fit the model
# model.fit(X, y,
#           nb_epoch=Max_No_Epochs,
#           batch_size = Batch_Size,
#           validation_split=validation_portion,
#           callbacks=callbacks_list,
#           verbose = 2,
#           shuffle=True)


# fits the model on batches with real-time data augmentation:
# model.fit_generator(traindatagen.flow(X_train, y_train, batch_size=Batch_Size),
#                     validation_data=validdatagen.flow(X_valid, y_valid,
#                                                       batch_size=int(math.floor(Batch_Size * validation_portion)) + 1),
#                     samples_per_epoch=len(X_train),
#                     nb_val_samples = len(X_valid),
#                     nb_epoch=Max_No_Epochs,
#                     callbacks=callbacks_list,
#                     verbose=2)

# Evaluate on test data
# scores = model.evaluate(X_test, y_test, verbose=2)
# print("Error: %.2f%%" % (100 - scores[1] * 100))


train_history = model.fit_generator(generator=train_gen_velocity, steps_per_epoch= Num_Batches_Per_Train_Epoch,
                    validation_data=valid_gen_velocity, validation_steps = n_valid,
                    epochs=Max_No_Epochs,
                    callbacks=callbacks_list,
                    verbose = 2)

model.save( model_dir + '/final_model.h5')

# Save history
import pickle
with open('train_history.pickle', 'wb') as handle:
    pickle.dump(train_history, handle)


# Evaluate on test data
scores = model.evaluate_generator(generator=test_gen_velocity, steps=Num_Batches_Per_Test_Epoch)
print("Error: %.2f%%" % (100 - scores[1] * 100))




