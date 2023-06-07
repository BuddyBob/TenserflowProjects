import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images_img, train_labels), (test_images_img, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']




train_images = train_images_img/255
test_images = test_images_img/255

print(test_images.shape)

#RELU: function whose result is 0 when the value is equal to or less than 0, result is the value itself when greater than 0
#SOFT MAX: converts final values to give probability of each (sum all values = 1)\
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
clothing_model = tf.keras.Sequential(
    [layers.Flatten(input_shape=(28,28)),layers.Dense(64,activation="relu"), layers.Dense(32, activation="relu") , layers.Dense(units=10)]
)

#addition to model
probability = tf.keras.Sequential(
    [clothing_model, layers.Softmax()]
)

#SparseCategoricalCrossentropy used for more than 2 categories
clothing_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

clothing_model.fit(
    train_images,
    train_labels,
    
    epochs=12,
    verbose=1,
    validation_split = 0.2,
    callbacks=[callback]
)

clothing_model.save('clothing.h5')

out = probability.predict(test_images)
#probabilities


evaluated = clothing_model.evaluate(test_images, test_labels, verbose=1)
#evaluation
# print(evaluated)




# plt.figure(figsize=(12,12))
# for i in range(20):
#     plt.subplot(5,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images_img[i])
#     plt.xlabel(class_names[np.argmax(out[i])])
# plt.show()

