#!/usr/bin/env python
#training code based on https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
#keras version 2.2.4, tensorflow version 1.8.0

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K

import numpy as np
import random
import os

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

num_models_to_train = 10
for model_idx in range(num_models_to_train):
    np.random.seed(model_idx*100)
    random.seed(model_idx*100)
    print("On model",model_idx)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    model.save("model_"+str(model_idx)+".h5")
    presoftmax_model = Model(inputs=model.layers[0].input,
                             outputs=model.layers[-2].output)
    test_preds = presoftmax_model.predict(x_test)
    output_file = "preds_model_"+str(model_idx)+".txt"
    f = open(output_file, 'w')
    f.write("\n".join(["\t".join([str(x) for x in y])
                                     for y in test_preds]))
    f.close()
    os.system("gzip -f "+output_file)
    print('Test accuracy:', np.mean(np.argmax(test_presoftmax_preds,axis=-1)
                                    ==np.argmax(y_test,axis=-1)))
    
