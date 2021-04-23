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

def predict(input = {
    'character' : 0,
    'ascension' : 1,
    'floor' : 1,
    'hp' : 56,
    'gold' : 123,
    'path' : 'M|?|M|M|M|E|R|?|T|R|?|?|E|$|',
    'deck': 'Strike_G|Strike_G|Strike_G|Strike_G|Strike_G|Defend_G|Defend_G|',
    'relics': 'Ring of the Snake|Art of War|StoneCalendar|MawBank|Sundial',
    }):
 

 reloaded_model = tf.keras.models.load_model('path_classifier')
 input_dict = {name: tf.convert_to_tensor([value]) for name, value in input.items()}
 predictions = reloaded_model.predict(input_dict)
 prob = tf.nn.sigmoid(predictions[0])
 
 print(
"This particular path had a %.1f percent probability "
    "of winning." % (100 * prob))
 
 print(prob)
 return prob

 #model.predict();
  

def getInput():
    # get filepath
    # get file from path
    # set file to data frame
    # return dataframedata
    dataframe = pd.read_csv("testoutput.csv")

def getSuggestedPath(pathList = list()):
    pathsProbability = list()
    for path in pathList:
        pathsProbability.append(predict(path))

    prob = pathsProbability[0]
    
    for prob in pathsProbability:
        if prob > highestProb:
            highestProb = prob
    
    return highestProb
    

def train():
 # column order in CSV file
 column_names = ['character','ascension','floor','hp','gold','path','deck','relics','victory']

 dataframe = pd.read_csv("testoutput.csv")
 dataframe.info()
 dataframe.head()

 train, test = train_test_split(dataframe, test_size=0.2)
 train, val = train_test_split(train, test_size=0.2)
 print(len(train), 'train examples')
 print(len(val), 'validation examples')
 print(len(test), 'test examples')

 batch_size = 20
 train_ds = df_to_dataset(train, batch_size=batch_size)
 val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
 test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

 all_inputs = []
 encoded_features = []

# Numeric features.
 for header in ['character', 'ascension','floor', 'hp', 'gold']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)

# Categorical features encoded as string.
 categorical_cols = ['path','deck','relics']
 for header in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string')
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

 model.fit(train_ds, epochs=10, validation_data=val_ds)

 loss, accuracy = model.evaluate(test_ds)
 print("Accuracy", accuracy)

 model.save('path_classifier')

#------------------
predict()