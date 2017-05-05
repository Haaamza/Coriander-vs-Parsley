# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 23:40:52 2017

Classification of coriander (9zbr in Moroccan dialect) and parsley (Madnouss in Moroccan dialect)

@author: Redouane Lguensat

> inspired from Keras Blog: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
> Data is collected and provided by Ali Lakrakbi https://github.com/alilakrakbi/Coriander-vs-Parsley

"""
'''
Train: 464 images #137 for coriander and 327 for parsley
Validation: 170 images #77 coriander and 93 parsley

Data is unbalanced, this is a simple try with a shallow net, I am sure there is a large margin of improvement
through the use of class blalancing or juste sampling more from the coriander class

This is the directory structure:
```
data/
    train/
        coriander/
            001.jpg
            002.jpg
            ...
        parsley/
            001.jpg
            002.jpg
            ...
    validation/
        coriander/
            001.jpg
            002.jpg
            ...
        parsley/
            001.jpg
            002.jpg
            ...
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import regularizers



# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'dataset_coriander_parsley/train'
validation_data_dir = 'dataset_coriander_parsley/validation'
nb_train_samples = 464 #137 coriander and 327 parsley
nb_validation_samples = 170 #77 coriander and 93 parsley
epochs = 100
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(16, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)
                 , input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(2, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,#nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=500,#nb_validation_samples // batch_size,
    class_weight={0:1.0, 1:0.4}
    )

#model.save_weights('first_try.h5')

############## test
from keras.preprocessing.image import img_to_array, load_img

img = load_img('coriander_example.jpg', target_size=(img_width, img_height))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)
proba_result1 = model.predict(x/255)

img = load_img('parsley_example.jpg', target_size=(img_width, img_height))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)
proba_result2 = model.predict(x/255)