# Identifying Melanoma in images of skin lesions

The purpose of this project was to create a machine learning model that can accurately predict if based on its image, a skin lesion is cancerous (malignant) or non-cancerous (benign). The dataset was hosted on kaggle as tfrecord files. First, I created an efficient data input pipeline using Tensorflow, which allowed me to efficiently run experiments. Next, I created various data augmentation functions for my data input pipeline that effectively increased the size of my training dataset. Next Tensorflow and Keras were used to create Convolutional Neural Network ("CNN") machine learning models (both transfer learning models and models that were trained from scratch were incorporated into my experiments). For training, I used both Google Colaboratory, and Kaggle Notebooks, which allowed me to Application-Specific Integrated Circuit (ASIC) machines specifically designed for machine learning, Tensor Processing Units (TPU), which reduces the time it takes to conduct an experiment compared to using a Graphics Processing Unit, or my home laptop. After training, I used matplotlib pyplot to graph various metrics of my model. Once I am ready to make predictions, I used Tensorflow to pass my test dataset into my machine learning model and stored the output in a Pandas DataFrame that was exported to a CSV file.

## Questions:
1. Which pre-trained models work best with the dataset I am using?
2. What model architecture will maximize the area under the curve ("AUC" -- a metric for imbalanced data) for a model being trained from scratch?
3. Most importantly, what is the probability a given image of a skin lesion is malignant?

## Dataset
1. https://www.kaggle.com/c/siim-isic-melanoma-classification/data [1]

## Tasks

### Build data input pipeline

1. Identify tfrecord files located in the dataset cloud storage bucket.
2. Split tfrecord files into training and validation sets
3. Create tf.data.TFRecordDataset using tensorflow for training, validation and test datasets
4. Parse tf.train.Examples contained in tfrecord files
5. Normalize image pixel values
6. Apply data augmentation methods to effectively increase dataset size

### Create machine learning models
1. Create function for sets of layers that will perform convolutions (convolutional layer, Max Pooling layer, batch normalization layer, dense layer, dropout and activation layers as appropiate)
2. Create function for sets of layers that will perform deconvolutions (Conv2DTranspose, batch normalization, dense, and dropout), and experiment with deconvolution sets of layers after convolution layers
3. Create function that will create a model to be trained from scratch, and compile the model.
4. Create a function that will create a model that will leverage transfer learning, and compile the model.
5. Calculate and apply class weights and initial output bias due to the fact the dataset is unbalanced.
6. Experiment

### Training and monitoring performance
1. Create model by calling above function with the TPU strategy.
2. Train model using the .fit method of sequential or functional model.
3. Create function to plot various metrics on how the model performed on training and validation data, and use this function to monitor model performance during training, and check if the model is overfitting training data.

### Inference
1. Pass test data into model to get predictions
2. Obtain identifier of each test example in test dataset in the same order test data was input into model.
3. Create a Pandas DataFrame of predictions and identifiers of each test example
4. Save DataFrame as a CSV file. 

### Results
1. The best model trained from scratch scored 0.8461 AUC on unseen test data
2. The best model that used transfer learning used the Xception model and scored 0.9058 AUC on unseen test data









[1]:
>The ISIC 2020 Challenge Dataset https://doi.org/10.34970/2020-ds01 (c) by ISDIS, 2020
>
>Creative Commons Attribution-Non Commercial 4.0 International License.
>
>The dataset was generated by the International Skin Imaging Collaboration (ISIC) and images are from the following sources: Hospital Clínic de Barcelona, Medical >University of Vienna, Memorial Sloan Kettering Cancer Center, Melanoma Institute Australia, The University of Queensland, and the University of Athens Medical >School.
>
>You should have received a copy of the license along with this work.
>
>If not, see https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.
