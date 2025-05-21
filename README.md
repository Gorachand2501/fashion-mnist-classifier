# 👕 Fashion MNIST Clothing Classifier (TensorFlow/Keras)

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify images from the **Fashion MNIST** dataset — a more challenging alternative to the original MNIST digit classification task.

The dataset contains grayscale images of 10 clothing categories like **Shirts, Sneakers, Dresses, Ankle boots**, etc.

---

## ✅ Features

- 🧠 CNN model built with TensorFlow/Keras
- 👗 Classifies 10 types of clothing from 28x28 grayscale images
- 🧹 Includes **Dropout layers** to reduce overfitting
- 📈 Training/validation **loss & accuracy plots**
- 📊 **Confusion matrix** for error analysis
- 💾 Saves trained model as `fashion_mnist_cnn.h5`
- ☁️ Designed to run on **Google Colab**
- 🧪 Achieves ~**90%+ test accuracy**

---

## 🏃‍♂️ Run it on Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://github.com/Gorachand2501/mnist-digit-classifier/blob/main/MNIST_Digit_Calssification%20(CNN).ipynb)

## 📦 Requirements
- tensorflow
- keras
- matplotlib
- scikit-learn


## 🧠 Model Architecture

Input: 28x28x1 grayscale image  
Conv2D(32, 3x3) → ReLU  
→ MaxPooling(2x2)  
→ Conv2D(64, 3x3) → ReLU  
→ MaxPooling(2x2)  
→ Flatten  
→ Dense(128) → ReLU  
→ Dropout(0.5)  
→ Dense(10) → Softmax
