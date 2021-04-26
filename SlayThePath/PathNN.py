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
import glob

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()

  labels = dataframe.pop('victory')
  #labels = dataframe.pop('maxFloor')
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

def predict(reloaded_model, input = { 'character' : 3,'ascension' : 0, 'floor' : 1, 'hp' : 70, 'gold' : 218, 'path' : 'M|?|M|M|$|E|R|M|T|?|E|M|R|M|R|BOSS|', 'deck': 'Strike_G+1|Strike_G|Strike_G|Strike_G|Strike_G+1|Defend_G|Defend_G|Defend_G|Defend_G|Defend_G|Survivor|Neutralize|All Out Attack+1|After Image|Tactician|Deflect+1|Catalyst|Dagger Spray|Injury|Deadly Poison|Leg Sweep+1|Dash|Adrenaline|Poisoned Stab+1|Footwork|Shame|Dagger Throw+1|All Out Attack+1|Deflect|Flying Knee+1|PiercingWail+1|Terror|Adrenaline|CurseOfTheBell|Bouncing Flask|Riddle With Holes+1|Wraith Form v2+1|Backflip|Backflip|Parasite|Flying Knee+1|Backflip+1|Noxious Fumes+1', 'relics': 'Ring of the Snake|ClockworkSouvenir|MawBank|PreservedInsect|Golden Idol|Strawberry|Fusion Hammer|Molten Egg 2|Toolbox|Kunai|Lantern|Whetstone|Bag of Preparation|Calling Bell|Juzu Bracelet|Frozen Egg 2|TungstenRod|Mummified Hand|Pen Nib|Nunchaku|Bottled Tornado',}):
 
 #reloaded_model = tf.keras.models.load_model('path_classifier')
 input_dict = {name: tf.convert_to_tensor([value]) for name, value in input.items()}
 predictions = reloaded_model.predict(input_dict)
 #print(input_dict)
 #print(predictions)

 prob = tf.nn.sigmoid(predictions[0])
 
 print("This particular path had a %.1f percent probability of winning." % (100 * prob))
 
# print(prob)
 return prob

  

def getInput():
    # get filepath
    # get file from path
    # set file to data frame
    # return dataframedata
    dataframe = pd.read_csv("testoutput.csv")

def getSuggestedPath():
    potentialPathsDF = pd.read_csv("random_sampling.csv")
    potentialPathsDF.pop('victory')
    potentialPaths = potentialPathsDF.to_dict('records')
    pathsProbability = list()
    reloaded_model = tf.keras.models.load_model('path_classifier')

    for record in potentialPaths:
        pathsProbability.append(predict(reloaded_model, record))
   
    highestProb = pathsProbability[0]
    pathIndex = 0
    currentPathIndex = 0

    for prob in pathsProbability:
        if prob > highestProb:
            pathIndex = currentPathIndex
            highestProb = prob
        
        currentPathIndex += 1 

    #Get the path with the highest prob of winning
    currentPathIndex = 0
    pathResult = potentialPaths[0]
    for record in potentialPaths:
        if(currentPathIndex == pathIndex):
            pathResult = record
        
        currentPathIndex += 1


    print("Recommended path:")
    #pathResult = potentialPaths[0]
    print(pathResult['path'])
    #return highestProb
    

def train():
 # column order in CSV file
 column_names = ['character','ascension','floor','hp','gold','path','deck','relics','victory']

 #dataframe = pd.read_csv("testoutputVictory.csv")
 #dataframe2 = pd.read_csv("random_sampling2")
 csv_file_list = list()

 csv_file_list = ["testoutputVictory.csv", "random_sampling2.csv", "random_sampling_10_14_06.csv", "random_sampling_10_14_08.csv"]
 
 #list_of_dataframes = []
 #for filename in csv_file_list:
 #   list_of_dataframes.append(pd.read_csv(filename))
 #
 #dataframe = pd.concat(list_of_dataframes)
 

 path = r'D:\git2\Programming\SlayThePath\SlayThePath\SlayThePath\TrainingData' 
 all_files = glob.glob(path + "/*.csv")

 dataframeList = []
 for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    dataframeList.append(df)

 dataframe = pd.concat(dataframeList, axis=0, ignore_index=True)




 
 dataframe.info()
 dataframe.head()

 train, test = train_test_split(dataframe, test_size=0.2)
 train, val = train_test_split(train, test_size=0.2)
 print(len(train), 'train examples')
 print(len(val), 'validation examples')
 print(len(test), 'test examples')

 batch_size = 256
 train_ds = df_to_dataset(train, batch_size=batch_size)

 
 
 val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
 [(train_features, label_batch)] = val_ds.take(1)
 print('Every feature:', list(train_features.keys()))
 print('A batch of ascension:', train_features['ascension'])
 print('A batch of targets:', label_batch )

 test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
 [(train_features, label_batch)] = test_ds.take(1)
 print('Every feature:', list(train_features.keys()))
 print('A batch of ascension:', train_features['ascension'])
 print('A batch of targets:', label_batch )

 all_inputs = []
 encoded_features = []

# Numeric features.
 for header in ['character', 'ascension', 'hp', 'gold']:
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
 x = tf.keras.layers.Dense(6, activation="relu")(all_features)
 x = tf.keras.layers.Dropout(0.5)(x)
 output = tf.keras.layers.Dense(1)(x)
 model = tf.keras.Model(all_inputs, output)
 model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

 tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

 model.fit(train_ds, epochs=3, validation_data=val_ds)

 loss, accuracy = model.evaluate(test_ds)
 print("Accuracy", accuracy)

 model.save('path_classifier')

#------------------Debug
#reloaded_model = tf.keras.models.load_model('path_classifier')
#predict(reloaded_model)
#getSuggestedPath()
