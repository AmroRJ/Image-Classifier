🐱🐶 Image Classifier: Cats vs. Dogs
A deep learning project that builds and trains a Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to classify images of cats and dogs. This project demonstrates the complete end-to-end workflow of a computer vision task, from data preparation and preprocessing to model training, evaluation, and prediction.

📋 Project Overview
This project tackles the classic binary image classification problem. The model learns to distinguish between images of cats and dogs, achieving strong validation accuracy. The notebook is designed to be run on Google Colab and provides a clear, step-by-step guide to building a CNN.

✨ Features
End-to-End Pipeline: Covers every step from installing libraries to making predictions on new images.
Data Preprocessing: Includes robust handling of corrupt images and utilizes Keras' ImageDataGenerator for efficient data loading and augmentation (rescaling, train/validation split).
CNN Architecture: Implements a classic CNN structure with convolutional layers, max-pooling, dropout for regularization, and dense layers.
Training with Callbacks: Employs EarlyStopping to prevent overfitting and optimize training time.
Performance Visualization: Plots training and validation accuracy/loss curves to analyze model performance and learning behavior.
Model Evaluation: Calculates final accuracy on the validation set and provides a function (predict_image) to test the model on individual images.
Model Persistence: Trained model is saved as an HDF5 file (cat_dog_classifier_model.h5) for future use and deployment.

🛠️ Tech Stack
Language: Python
Deep Learning Framework: TensorFlow / Keras
Libraries: OpenCV, Matplotlib, PIL (Pillow), NumPy
Environment: Google Colab (Jupyter Notebook environment)

📁 Dataset
The model is trained on a subset of the famous Dogs vs. Cats dataset from Kaggle, expected to be in a directory structure like:
text
/content/data/PetImages/
    ├── Cat/
    └── Dog/
The notebook includes a crucial step to scan for and remove corrupt image files that could disrupt the training process.

📊 Model Architecture
The designed Convolutional Neural Network (CNN) has the following structure:
Conv2D Layer (32 filters, 3x3 kernel, ReLU activation)
MaxPooling2D Layer (2x2 pool size)
Conv2D Layer (64 filters, 3x3 kernel, ReLU activation)
MaxPooling2D Layer (2x2 pool size)
Flatten Layer
Dropout Layer (50% rate) to reduce overfitting
Dense Layer (64 units, ReLU activation)
Output Dense Layer (1 unit, Sigmoid activation for binary classification)
The model is compiled with the adam optimizer and binary_crossentropy loss function.

📈 Results
After training for 8 epochs (stopped early by the callback), the model achieved:
Training Accuracy: ~95%
Validation Accuracy: ~80%
The plot of accuracy shows the model learning effectively, though some overfitting is observed as training accuracy continues to rise while validation accuracy plateaus.
