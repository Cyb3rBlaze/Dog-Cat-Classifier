import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(256, 256, 3)))

model.add(tf.keras.layers.Conv2D(16, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.load_weights("training_1/cp.ckpt")

evaluate_image=Image.open("../output/dog.jpg")
evaluate_image=evaluate_image.resize((256, 256), Image.ANTIALIAS)
evaluate_image=np.array(evaluate_image).reshape((1, 256, 256, 3))
value=model.predict(evaluate_image)

print(value)

if value[0][0] > 0.5:
    print("The image is a dog")
else:
    print("The image is a cat")