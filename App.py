#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json
import os

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Load the pre-trained model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_leaf_disease_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

# Function to handle button click event
def classify_image():
    image_path = filedialog.askopenfilename()
    if image_path:
        image = Image.open(image_path)
        resized_img = image.resize((200, 200))
        img_tk = ImageTk.PhotoImage(resized_img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        prediction = predict_image_class(model, image_path, class_indices)
        result_label.config(text=f'Predicted Plant Disease: {prediction}')

# Create the main window
root = tk.Tk()
root.title('Plant Disease Predictor 🌿🔍')

# Set the size of the main window
root.geometry("600x400")

# Create a Canvas widget as the background with an image
background_image = Image.open("Template.jpg")  # Change "background_image.jpg" to your image file
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Create a button to browse images
browse_button = tk.Button(root, text="Browse image", command=classify_image, bg="white")
browse_button.place(relx=0.5, rely=0.5, anchor="center")

# Create a label to display the uploaded image
image_label = tk.Label(root, bg="white")
image_label.place(relx=0.5, rely=0.4, anchor="center")

# Create a label to display the prediction result
result_label = tk.Label(root, text="", bg="white")
result_label.place(relx=0.5, rely=0.9, anchor="center")

root.mainloop()

