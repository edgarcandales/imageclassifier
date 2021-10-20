from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import mnist

img_width, img_height = 28, 28

model = load_model("label_number.h5",compile = True)

# Load EMNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Reshape data
input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Cast numbers to float32
input_train = input_train.astype('float32')

# Scale data
input_train = input_train / 255

samples_to_predict = input_train

predictions = model.predict([samples_to_predict])
print(predictions[0])

# print("done")

