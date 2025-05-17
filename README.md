# 📱🌐 Neural Network SMS Text Classifier
A deep learning model built with TensorFlow, Keras, pandas, and NumPy that classifies SMS messages as ham (a normal message sent by a friend) or spam (an advertisement or a message sent by a company).

> 🧠 **This challenge was provided by [freeCodeCamp’s Machine Learning with Python course](https://www.freecodecamp.org/learn/machine-learning-with-python/).**

## 🛠 What I Did

- Preprocessed the training data by:
    - Converting text labels ("ham", "spam") to numeric labels (0, 1)
    - Using `Tokenizer` to convert the messages into sequences of integers representing words
    - Padding sequences to a uniform length for model input
- Built and trained a Sequential model
- Created a `predict_message` function to classify new SMS inputs

## 🤔 What I Learned

- **Handling out-of-vocabulary (OOV) words**: By specifying an `oov_token` in the tokenizer, any word not seen during training is mapped to a special `<OOV>` index instead of being ignored or causing an error. This ensures that the model can still process new or misspelled words at run time, helping it generalize better to unseen text.
- **Implementing a Bidirectional LSTM**: During testing, there was one message in particular that kept failing the test function. So, I decided to explore changing the layers of the Sequential model to improve accuracy, and found out about Bidirectional LSTMs. A regular LSTM reads text left-to-right. A Bidirectional LSTM reads the message both forward and backward, so it can understand the full context of every word. Adding this layer ultimately made the model much more accurate.
- **Sigmoid output**: The model ends with one output neuron using a sigmoid function, which gives a number between 0 and 1. This number is the model’s confidence that a message is spam. If it’s 0.5 or higher, it’s spam; if it’s lower, it’s ham.

## 🚀 Future Improvements

- **Pre-Trained Word Embedding**: Instead of learning word meanings from scratch, using pretrained embeddings like GloVe or Word2Vec could improve accuracy.
- **Explore Other Architecture**: I want to try even more types of layers or model structures (like GRUs or transformers) and compare results.
- **Hyperparameter Tuning**: Adjusting different hyperparameters such as the learning rate, batch size, etc., could lead to better performance.
