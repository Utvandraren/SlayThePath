import Window as win

#win.constructWindow()
import os
#import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pathlib


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

#print("TensorFlow version: {}".format(tf.__version__))
#print("Eager execution: {}".format(tf.executing_eagerly()))

# column order in CSV file
column_names = ['ascension_level', 'character_chosen', 'neow_bonus', 'path_taken', 'victory']
feature_names = column_names[:-1]
label_name = column_names[-1]


class_names = ['False', 'True']

batch_size = 571
#filepath = os.path.abspath("TrainingDataWithOutput.csv")

dataframe = pd.read_csv("TrainingDataWithOutput.csv")
dataframe.info()

# In the original dataset "4" indicates the pet was not adopted.
#dataframe['target'] = np.where(dataframe['victory']== 'False', 'True')
# Drop un-used columns.
#dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


"""
train_dataset = tf.data.experimental.make_csv_dataset(
    filepath,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,)

test_dataset = train_dataset
"""
#train_dataset = all_dataset.skip(50)


#features, labels = next(iter(train_dataset))
#print(features)
#for element in train_dataset:
#  print(element)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('victory')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

# convert column "a" to int64 dtype and "b" to complex type
#dataframe = dataframe.astype({"neow_bonus":  str, "character_chosen":  str, "path_taken": str})
dataframe = pd.Series(['path_taken', 'character_chosen', 'neow_bonus'], dtype="string")

#dataframe.dtypes()

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch )

def get_normalization_layer(name, dataset):
  # Create a Normalization layer for our feature.
  normalizer = preprocessing.Normalization()

  # Prepare a Dataset that only yields our feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a StringLookup layer which will turn strings into integer indices
  if dtype == 'string':
    index = preprocessing.StringLookup(max_tokens=max_tokens)
  else:
    index = preprocessing.IntegerLookup(max_values=max_tokens)

  # Prepare a Dataset that only yields our feature
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Create a Discretization for our integer indices.
  encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

  # Prepare a Dataset that only yields our feature.
  feature_ds = feature_ds.map(index)

  # Learn the space of possible indices.
  encoder.adapt(feature_ds)

  # Apply one-hot encoding to our indices. The lambda function captures the
  # layer so we can use them, or include them in the functional model later.
  return lambda feature: encoder(index(feature))

"""
all_inputs = []
encoded_features = []

# Numeric features.
for header in ['ascension_level']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_dataset)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)

# Categorical features encoded as string.
categorical_cols = ['character_chosen', 'neow_bonus', 'path_taken']
for header in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(header, train_dataset, dtype='string')
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)



all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])


tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

model.fit(train_dataset, epochs=10, validation_data=test_dataset)
"""
#from tensorflow.keras import models
#from tensorflow.keras import layers
#network = models.Sequential()
#network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
#network.add(layers.Dense(10, activation='softmax'))

#network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#train_images = train_images.reshape((60000, 28 * 28))
#train_images = train_images.astype('float32') / 255
#test_images = test_images.reshape((10000, 28 * 28))
#test_images = test_images.astype('float32') / 255

#from tensorflow.keras.utils import to_categorical
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)

#network.fit(train_images, train_labels, epochs=5, batch_size=128)