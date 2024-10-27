
# Fashion MNIST Classification with Keras

    The goal of this project is to build a multiclass classification model using Keras with TensorFlow to classify fashion items from the Fashion MNIST dataset.



# Dataset description

    The Fashion MNIST dataset consists of 70,000 grayscale images of 10 fashion categories, each of size 28x28 pixels. The dataset is divided as follows:

    Training set: 60,000 images
    Testing set: 10,000 images
    Each image belongs to one of 10 classes:

    T-shirt/top
    Trouser
    Pullover
    Dress
    Coat
    Sandal
    Shirt
    Sneaker
    Bag
    Ankle boot



## Steps to Run the Code in Jupyter Notebook

Load the Dataset

    Import the necessary libraries.
    Load the Fashion MNIST dataset using keras.datasets.
    Split the dataset into training and testing sets.

Preprocess the Data

    Normalize the images by scaling pixel values between 0 and 1.
    Flatten the images from 28x28 into 1D vectors of size 784.
    Visualize a few sample images with their corresponding labels.

Build the Neural Network Model

    Model Creation: Create a sequential model using Keras with the following structure:
        An input layer that accepts a 784-dimensional vector.
        Two dense hidden layers with 128 and 64 neurons, batch normalization, ReLU activation, and dropout for regularization.
        An output layer with 10 neurons and softmax activation for multiclass classification.

Compile the Model

    Choose the optimizer (e.g., Adam).
    Set the loss function to sparse_categorical_crossentropy (suitable for multiclass classification).
    Define evaluation metrics (e.g., accuracy).

Train the Model

    Fit the model on the training data, using an 80-20 split for training and validation.
    Monitor loss and accuracy using the validation data.
    Apply EarlyStopping and a learning rate scheduler to prevent overfitting and enhance training stability.

Evaluate the Model

    Calculate and print the following metrics:
        Accuracy on the test set
        Confusion Matrix: Visualize with a heatmap to understand class-wise performance.
        Classification Report: Generate precision, recall, and F1-score for each class.

Visualizations:

    Plot training and validation loss curves.
    Plot training and validation accuracy curves.
    Display a bar graph of precision, recall, and F1-score for each class.
## Dependencies and Installation Instructions
    bash

    pip install tensorflow numpy pandas matplotlib scikit-learn seaborn

## Example Output

    After training the neural network, you will obtain metrics to evaluate the model's performance. The visualizations of the loss and accuracy will provide insights into the learning process and help assess how well the model generalizes to unseen data.
