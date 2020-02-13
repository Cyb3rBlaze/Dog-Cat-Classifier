import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from time import time
from tensorflow.keras.callbacks import TensorBoard

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        "../data/train",
        target_size=(64, 64),
        batch_size=20,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        "../data/test",
        target_size=(64, 64),
        batch_size=5,
        class_mode='binary')

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(64, 64, 3)))

model.add(tf.keras.layers.Conv2D(5, 3, activation='tanh'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None))

model.add(tf.keras.layers.Conv2D(5, 3, activation='tanh'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='tanh'))
model.add(tf.keras.layers.Dense(84, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=test_generator,
      validation_steps=50,
      verbose=1,
      callbacks=[cp_callback])