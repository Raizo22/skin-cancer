import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your saved model
model = load_model(r"E:\skin cancer project\cancer_model_cnn_no_smote (3).h5")

# Assuming classes are in the order specified in your training script
class_labels = {0: 'normal', 1: 'benign', 2: 'malignant'}

# Function to preprocess and predict a single image
def predict_single_image(img):
    img = img.resize((128, 128))  # Resize image
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image data if needed

    # Make predictions
    predictions = model.predict(img_array)

    # Assuming classes are in the order specified in your training script
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = np.max(predictions)  # Get confidence of the prediction

    return predicted_label, confidence, img_array[0]

# Define Streamlit interface
def classify_skin_cancer(image_file):
    img = Image.open(image_file)
    predicted_label, confidence, processed_img = predict_single_image(img)
    st.write("Prediction:", predicted_label)
    st.write("Confidence:", f"{confidence:.2f}")
    st.image(processed_img, caption='Processed Image', use_column_width=True)

st.title("Skin Cancer Classifier")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    classify_skin_cancer(uploaded_file)

