import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', names=['mpg', 'cyl' ,'disp', 'hp', 'weight', 'acc' ,'year', 'origin','name'],  na_values="?", comment="\t", sep=" ", skipinitialspace=True)
del df['name']
df = df.dropna()
# x = df[0:-1]
x = df.iloc[:,1:]
y = df['mpg']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

normalization = tf.keras.layers.Normalization()
normalization.adapt(x_train)


mpg_model = tf.keras.Sequential(
    [normalization,layers.Dense(64,activation="relu"), layers.Dense(32, activation="relu") , layers.Dense(units=1)]
)



mpg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mean_absolute_error'
    )

history = mpg_model.fit(
    #xtrain_ytrain
    x_train,
    y_train,
    #train times
    epochs=80,
    #dont print
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)


out = mpg_model.predict(x_test)
evaluated = mpg_model.evaluate(x_test, y_test, verbose=1)

df1 = pd.DataFrame(out, columns=["result"])
df1['mpg'] = df['mpg']
print(df1)
print(evaluated)


