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

