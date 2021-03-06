import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


# load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


# make model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(train_images, train_labels, epochs=5)

# validation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('test accuracy : ', test_acc)

# prediction
predictions = model.predict(test_images)
pred_0 = np.argmax(predictions[0])
print('1st test data label : ', test_labels[0], ', 1st test data predicted label : ', pred_0)

plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(test_images[test_labels[0]])

plt.subplot(2, 1, 2)
plt.imshow(test_images[pred_0])

plt.show()
