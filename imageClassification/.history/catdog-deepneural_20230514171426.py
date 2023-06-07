import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)


train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

batch = train_dataset.take(1)

# catdog_model = tf.keras.Sequential([
#   layers.Rescaling(1./255, input_shape=(160,160,3)), layers.Flatten(), layers.Dense(64), layers.Dense(32), layers.Dense(units=1), layers.Softmax()
# ])


catdog_model = tf.keras.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.Rescaling(1./255, input_shape=(160,160,3)),
  layers.Flatten(), 
  layers.Dense(64), 
  layers.Dense(32), 
  layers.Dense(units=1), 
  layers.Softmax()
])



catdog_model.compile(
  optimizer = tf.keras.optimizers.Adam(),
  loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
  metrics = ["Accuracy"]
)

history = catdog_model.fit(
  train_dataset,
  epochs = 4,
  validation_data = validation_dataset,
  verbose = 1
)

catdog_model.save("catdog_cnn_model.h5")

# catdog_model = keras.models.load_model("catdog_model.h5")

predicted = catdog_model.predict(train_dataset.take(1))
print(predicted)
plt.figure(figsize=(10, 10))
for image_lst, label_lst in batch:
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_lst[i].numpy().astype("uint8"))
    plt.title(label_lst[i].numpy())
    plt.axis("off")
# plt.show()