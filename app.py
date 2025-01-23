# Streamlit app code

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load the model
loaded_model = tf.keras.models.load_model('efficientnetb1_model.h5')

def predict_image_with_heatmap(img):
    print("Predicting image...")
    
    # Resize the image to (256, 256) and convert to a 3D array
    img_copy = img.copy()
    img_3d = cv2.resize(img_copy, (256, 256))
    img_3d = np.array(img_3d).reshape(-1, 256, 256, 3)

    # Predict using the loaded model
    prediction = loaded_model.predict(img_3d)[0]

    class_labels = ["No Cancer", "Adenocarcinoma", "Squamous Cell Carcinoma"]

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    
    if predicted_class_index < len(class_labels):
        predicted_class = class_labels[predicted_class_index]
        confidence_percentage = prediction[predicted_class_index] * 100
        
        # If confidence percentage is above 99.96%, show prediction, else return "Wrong Image"
        if confidence_percentage > 99.96:
            # Get the output of the last convolutional layer in the model
            last_conv_layer = loaded_model.layers[0].get_layer("top_activation")
            last_conv_layer_model = tf.keras.Model(loaded_model.layers[0].input, last_conv_layer.output)
            with tf.GradientTape() as tape:
                conv_outputs = last_conv_layer_model(img_3d)
                loss = tf.reduce_mean(conv_outputs[:, :, :, predicted_class_index])
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()[0]
            
            # Apply heatmap to the original image
            heatmap = cv2.resize(heatmap, (img_copy.shape[1], img_copy.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(heatmap, 0.5, img_copy, 0.5, 0)
            
            # Convert heatmap to RGB format for display
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            return heatmap_rgb, f"Predicted: {predicted_class} (Confidence: {confidence_percentage:.2f}%)"
        else:
            return None, "Wrong Image"
    else:
        return None, "Wrong Image"

# Streamlit app
st.title("EfficientnetB1 Model for Cancer Detection using biopsy images")
st.write("Upload an image to detect cancer and visualize the heatmap.")

# File uploader on the main screen
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Display the original image
    st.subheader("Uploaded Image")
    st.image(img, caption='Original Image', use_column_width=True)
    
    # Add a progress bar for prediction
    with st.spinner("Analyzing the image..."):
        # Predict and display the heatmap and label
        heatmap, label = predict_image_with_heatmap(img)
    
    if heatmap is not None:
        st.subheader("Heatmap Visualization")
        st.image(heatmap, caption='Heatmap Overlay', use_column_width=True)
    
    st.subheader("Prediction Result")
    st.write(label)
else:
    st.info("Please upload an image to get started.")