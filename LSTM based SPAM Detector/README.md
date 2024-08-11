# Spam Detection with LSTM

This project demonstrates the implementation of a spam detection model using an LSTM (Long Short-Term Memory) network. The model is trained on a dataset of SMS messages to classify them as either spam or ham (non-spam).

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training Results](#training-results)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build a binary classification model that can accurately predict whether a given SMS message is spam or ham. The model is built using TensorFlow and Keras, and it leverages an LSTM layer to capture the sequential nature of text data.

## Installation

To get started with the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/spam-detection-lstm.git
    cd spam-detection-lstm
    ```

2. **Install the required packages**:
    This project requires TensorFlow, NumPy, pandas, and scikit-learn. You can install the necessary packages using pip:

    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib
    ```

3. **Download the dataset**:
    The dataset used in this project is the SMS Spam Collection dataset. It is automatically downloaded using `wget` in the Jupyter notebook provided.

## Usage

1. **Training the model**:
    The project includes a Jupyter notebook where you can train the model. Run the notebook to train the model on the dataset and evaluate its performance.

    ```bash
    jupyter notebook spam_detection.ipynb
    ```

2. **Making predictions**:
    After training the model, you can use it to predict whether new SMS messages are spam or ham. Use the following code to predict a specific message:

    ```python
    sentences = ["Your free entry in the contest is waiting!", "I'll call you back later."]
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=T)
    predictions = model.predict(padded_sequences)
    ```

    The model will output probabilities for each message being spam. You can threshold these probabilities to classify the messages.

## Model Architecture

The LSTM model used in this project has the following architecture:

- **Embedding Layer**: Converts words to dense vectors of fixed size.
- **LSTM Layer**: Captures sequential dependencies in the text.
- **GlobalMaxPooling1D Layer**: Reduces each feature map to a single number by taking the maximum value.
- **Dense Layer**: Outputs the probability of the message being spam using a sigmoid activation function.

## Dataset

The dataset used in this project is the [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset). It contains a collection of SMS messages labeled as `spam` or `ham`. The dataset is used to train and evaluate the LSTM model.

## Training Results

During training, the model's performance is tracked using accuracy and loss metrics. Below are the training results for 10 epochs:

- **Final Training Accuracy**: 99.79%
- **Final Validation Accuracy**: 98.15%
- **Final Training Loss**: 0.0464
- **Final Validation Loss**: 0.0945

The model achieves a high level of accuracy in detecting spam messages.

## Prediction

To make predictions with the trained model, use the following steps:

1. Prepare the message(s) you want to predict.
2. Tokenize and pad the message(s).
3. Pass the preprocessed message(s) through the model to get a spam probability.

Refer to the example code provided in the notebook or the [Usage](#usage) section above.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
