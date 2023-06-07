import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', names=['mpg', 'cyl' ,'disp', 'hp', 'weight', 'acc' ,'year', 'origin','name'],  na_values="?", comment="\t", sep=" ", skipinitialspace=True)
# print(df.head(5))
# print(df.iloc[354])
del df['name']
df = df.dropna()
x = df[0:-1]
y = df['mpg']

normalization = tf.keras.layers.Normalization(axis=-1)
normalization.adapt(x)


mpg_model = tf.keras.Sequential(
    [normalization, layers.Dense(units=1)]
)

mpg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.15),
    loss='mean_absolute_error')

history = mpg_model.fit(
    #xtrain_ytrain
    x,
    y,
    #train times
    epochs=80,
    #dont print
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)


out = mpg_model.predict(x)
print(out)

df1 = pd.DataFrame(out, columns=["result"])
df1['mpg'] = df['mpg']
print(df1)




