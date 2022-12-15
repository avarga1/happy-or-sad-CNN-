# Happy or Sad Image Classification
This code is a machine learning program that trains a model to classify images as happy or sad. The model is built using the Tensorflow library.

## Getting Started
These instructions will help you set up and run the code on your local machine.

### Prerequisites
You will need to have the following libraries installed to run this code:

-matplotlib

-numpy

-os

-tensorflow

### Running the code

To run the code, you will need to have the happy and sad images stored in separate directories called 
"happy" and "sad" in a directory called "data" in the same location as the code file.

To run the code, simply execute it using Python:
```
python happy_or_sad.py
```

## Code Structure
The code is structured as follows:

-Import required libraries, including **'matplotlib'*** for plotting images, **'numpy'** for numerical operations, 
**'os'** for interacting with the file system, and **'tensorflow'** for building and training the model.
It also imports the **'ImageDataGenerator'** class from the Tensorflow Keras library, which will be used to load and preprocess the images.
Define the directories where the happy and sad images are stored.
Print out a sample happy and sad image using the **'load_img'** function from the Tensorflow Keras library.
Load a sample image, convert it into a numpy array using the **'img_to_array'** function, and print out the shape and maximum pixel value.
Define a custom callback class called **'myCallback'**, which is used to stop training the model once a certain level of accuracy is reached.
Define a function called **'image_generator'** that uses the **'ImageDataGenerator'** class to create an object that can load and preprocess the images.
Define a function called **'train_happy_sad_model'** that builds and trains the model.
The model is a simple convolutional neural network with three convolutional layers, each followed by a max pooling layer,
followed by a dense layer with 512 units and a sigmoid activation function. 
The model is compiled using the RMSprop optimizer and binary crossentropy loss function,
and is trained using the **'fit'** method and the **'myCallback'** class defined earlier. The function returns the training history.

