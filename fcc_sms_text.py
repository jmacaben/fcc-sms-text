# Some of the given code may have been changed for use outside of a notebook environment

# Cell 1 (given)
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Cell 2 (given)
import os
import requests

train_url = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
test_url = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

def download_file(url, path):
    if not os.path.exists(path):
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded '{path}'")
    else:
        print(f"'{path}' already exists. Skipping download.")

download_file(train_url, train_file_path)
download_file(test_url, test_file_path)

# Cell 3
# Load dataset
# Each row has a message and a label (ham or spam)
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# Convert text labels to numeric labels
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})

# Create a tokenizer to convert words to integers
# oov_token: handles words not seen in training by replacing them with "<OOV>"
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")

# Build vocabulary from training messages
tokenizer.fit_on_texts(train_data['message'])

vocab_size = len(tokenizer.word_index) + 1

# Convert messages to sequences and pad them
# Basically replaces each word in every SMS with its corresponding number.
# Also, pad sequences so they are all the same length
max_length = 100
tr_sequences = tokenizer.texts_to_sequences(train_data['message'])
tr_padded = tf.keras.preprocessing.sequence.pad_sequences(tr_sequences, maxlen=max_length, padding='post')

te_sequences = tokenizer.texts_to_sequences(test_data['message'])
te_padded = tf.keras.preprocessing.sequence.pad_sequences(te_sequences, maxlen=max_length, padding='post')

# Convert labels to NumPy arrays
tr_labels = np.array(train_data['label'])
te_labels = np.array(test_data['label'])

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)), 
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(tr_padded, tr_labels, epochs=10, validation_data=(te_padded, te_labels), verbose=2)

# Function to predict messages
def predict_message(pred_text):
    seq = tokenizer.texts_to_sequences([pred_text]) # Convert input message to sequence of integers
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, padding='post') # Pad it to match the training input shape
    pred = model.predict(padded)[0][0]
    label = 'spam' if pred >= 0.5 else 'ham'
    return [float(pred), label]

# Cell 4
# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
