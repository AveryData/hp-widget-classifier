# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:53:23 2023

@author: avery
"""

import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import shap
import matplotlib  as plt
import matplotlib.pyplot as pl
import streamlit.components.v1 as components
import os
import random
from streamlit_image_select import image_select


# Load in the model
model = tf.keras.models.load_model('casting_product_detection.hdf5')
image_shape = (512,512,1) 

# Set the title of the app
st.title("Widget Inspector Tool")

# Define the paths to your two sample images
image1_path = "Image/1.png"
image2_path = "Image/2.png"

# Load the images using PIL
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# Display the images in the app
st.image([image1, image2], caption=["Defect", "Normal"], width=300)

# Add a file uploader component to the app
st.subheader("Choose your widget image to get a classification: :point_down:")
#uploaded_file = st.file_uploader("Choose your widget image to get a classification: :point_down:")


# Define paths
test_path = "images/casting_512x512/casting_512x512/"

# ok_numbers = ['1344', '935', '9920', '1421','2844', '1213']
# def_numbers = ['2783', '2486', '112', '534', '40', '1683']

ok_numbers = ['1344', '222', '4630', '7664','8875', '9818']
def_numbers = ['255', '1187', '7248', '5545', '5696', '9683']
all_names = ok_numbers + def_numbers

okay_images = [ test_path + 'ok_front/cast_ok_0_' + i + '.jpeg' for i in ok_numbers]
def_images = [ test_path + 'def_front/cast_def_0_' + i + '.jpeg' for i in def_numbers]
all_images = okay_images + def_images

# Dropdown with images 
img_selected = image_select(
    label="",
    images= all_images,
    captions=all_names,
)

# If an image is uploaded, display it in the app
if img_selected is not None:
    uploaded_image = Image.open(img_selected)
    uploaded_image.save('temp.jpg')
    # Test the image for prediction
    
    predicted_label = "?"
    img_pred = cv2.imread('temp.jpg', cv2.IMREAD_GRAYSCALE)
    img_pred = img_pred / 255 # rescale
    img_pred = img_pred.reshape(1, *image_shape) # reshape
    
    prediction = model.predict(img_pred)
    if (prediction < 0.5):
        predicted_label = "Defect"
        prob = round((1-prediction.sum()) * 100,1)
        phrase = 'Widget is  :red[DEFECT] at {}% confidence'.format(prob)
        st.header(phrase)

    # Predicted Class : OK
    else:
        predicted_label = "Okay"
        prob = round(prediction.sum() * 100,0)
        phrase = 'Widgets is  :green[OKAY] at {}% confidence'.format(prob)
        st.header(phrase)
        
        
     # Show the image    
    st.image(uploaded_image, caption=predicted_label + ' Widget', width=300)
    
    # Do the SHAP explainer 
    n = 100
    random.seed(10)
    ok_files = random.sample(os.listdir(test_path + "ok_front"), n )
    ok_files = [test_path + "ok_front/" + i  for i in ok_files]
    def_files = random.sample(os.listdir(test_path +  "def_front"), n )
    def_files = [test_path +  "def_front/" + i  for i in def_files]
    
    test_files = [ok_files + def_files ][0]
    
    X = [cv2.imread(i,cv2.IMREAD_GRAYSCALE).reshape(1,*image_shape)  / 255 for i in test_files]
    
    m = 142
    
    explainer = shap.DeepExplainer(model,X[100]) 
    shap_values = explainer.shap_values(img_pred)
    f = shap.image_plot(shap_values, img_pred, show=False) 
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(f, bbox_inches='tight',dpi=300,pad_inches=0)
    pl.clf()
    


    
    