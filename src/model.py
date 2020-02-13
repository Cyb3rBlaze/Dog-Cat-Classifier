import tensorflow as tf
from PIL import Image
import numpy as np

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

model.load_weights("training_1/cp.ckpt")

evaluate_image=Image.open("../output/dog.jpg")
evaluate_image=evaluate_image.resize((64, 64), Image.ANTIALIAS)
evaluate_image=np.array(evaluate_image).reshape((1, 64, 64, 3))
value=model.predict(evaluate_image)

print(value)

if value[0][0] > 0.5:
    print("The image is a dog")
else:
    print("The image is a cat")