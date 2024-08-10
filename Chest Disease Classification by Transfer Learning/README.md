# Multi-Class Classification of Chest Diseases
This project is a multi-class classification system for chest diseases, specifically targeting the following classes:
- COVID-19
- Viral Pneumonia
- Bacterial Pneumonia
- Normal (No disease)

Due to the scarcity of the dataset, this project leverages the benefits of transfer learning. I utilize ResNet50, a pre-trained network on the ImageNet dataset, as the base model.

## Project Overview

### Model Architecture

1. **Base Model**: ResNet50 pre-trained on ImageNet.
2. **Modifications**:
    - The top of ResNet50 is removed.
    - The network parameters are frozen except for the last layers.
    - Two fully connected dense layers are added on top:
        - First dense layer with 256 neurons.
        - Second dense layer with 128 neurons.

### Why Transfer Learning?

Transfer learning allows us to utilize the knowledge gained from training on a large dataset (ImageNet) and apply it to our specific problem of chest disease classification. This is particularly useful 
when working with a limited dataset, as it helps improve the model's performance and reduces the risk of overfitting.

## Dataset

The dataset used in this project includes chest X-ray images categorized into four classes:
- COVID-19
- Viral Pneumonia
- Bacterial Pneumonia
- Normal

Due to the limited availability of labeled data, the transfer learning approach with ResNet50 is employed to achieve better classification accuracy.

## Steps
1. Import Libraries and Dataset

```
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# Specify training data directory
XRay_Directory = '/content/drive/MyDrive/Chest_X_Ray/train'
os.listdir(XRay_Directory)
```
2. Data Preparation and Augmentation
```
# Use image generator to generate tensor images data and normalize them
# Use 20% of the data for cross-validation
image_generator = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

# Generate batches of 32 images
train_generator = image_generator.flow_from_directory(batch_size = 32, directory = XRay_Directory, shuffle = True, target_size = (256, 256), class_mode = 'categorical', subset = 'training')

validation_generator = image_generator.flow_from_directory(batch_size = 32, directory = XRay_Directory, shuffle = True, target_size = (256,256), class_mode = 'categorical', subset = "validation")
```
3. Data Visualization
```
# Create a grid of 16 images along with their corresponding labels
label_names = {0 : 'Covid-19', 1 : 'Normal' , 2: 'Viral Pneumonia', 3 : 'Bacterial Pneumonia'}

L = 4
W = 4
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(train_images[i])
    axes[i].set_title(label_names[np.argmax(train_labels[i])])
    axes[i].axis('off')

plt.subplots_adjust(wspace = 0.5)
```
4. Load Pre-trained ResNet50 Model
```
base_model = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (256, 256, 3)))
base_model.summary()

# Freezing the base model layers
for layer in base_model.layers[:-10]:
    layer.trainable = False
```
5. Build and Train the Model
```
head_model = base_model.output
head_model = tf.keras.layers.AveragePooling2D(pool_size = (4, 4))(head_model)
head_model = Flatten(name = 'flatten')(head_model)
head_model = Dense(256, activation = 'relu')(head_model)
head_model = Dropout(0.3)(head_model)
head_model = Dense(128, activation = 'relu')(head_model)
head_model = Dropout(0.2)(head_model)
head_model = Dense(4, activation = 'softmax')(head_model)

model = Model(inputs = base_model.input, outputs = head_model)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(learning_rate = 1e-4, weight_decay=1e-6), metrics= ["accuracy"])

# Using early stopping and model checkpointing
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

history = model.fit(train_generator, steps_per_epoch= train_generator.n // 4, epochs = 50, validation_data= val_generator, validation_steps= val_generator.n // 4, callbacks=[checkpointer, earlystopping])
```
## Results

The model training includes multiple epochs, with early stopping and checkpointing to save the best model based on validation loss.

## Conclusion

This project demonstrates how to utilize transfer learning with the ResNet50 model for classifying chest X-ray images into different categories. 
The trained model can achieve high accuracy, provided a well-prepared dataset and proper augmentation techniques.
