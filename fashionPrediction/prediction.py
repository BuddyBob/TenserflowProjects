import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
clothing_model = load_model('clothing.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

clothes = []
for item in os.listdir('./items'):
    clothes.append(cv2.imread(f'./items/{item}'))
    
new_clothes = []
mask_l = []
for cloth in clothes:
    #manipulate image

    gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    #find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #create a mask to remove background
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, 1)
    # Add a border to the mask
    bordered = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    bordered_mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    masked_image = cv2.bitwise_and(bordered, bordered, mask=bordered_mask)
    #resize image
    resized = cv2.resize(bordered, (28,28), cloth, interpolation=cv2.INTER_AREA)
    new_clothes.append(resized)
    mask_l.append(resized)
    

probability = tf.keras.Sequential(
    [clothing_model, layers.Softmax()]
)

# turn new_clothes into numpy array
out = probability.predict(np.array(new_clothes))

plt.figure(figsize=(8,8))
for i in range(len(clothes)):
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(mask_l[i])
    plt.xlabel(class_names[np.argmax(out[i])])
plt.show()
plt.close()
