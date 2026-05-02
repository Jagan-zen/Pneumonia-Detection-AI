import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import joblib


# --- CONFIGURATION ---
st.set_page_config(page_title="Pneumonia Detection AI", layout="wide")
st.title("🫁 Pneumonia Detection & Explainability Dashboard")
st.markdown("""
This AI tool uses a **DenseNet121** model to detect Pneumonia in Chest X-rays. 
It includes **Grad-CAM** to show exactly which parts of the lungs the AI is analyzing.
""")


# --- LOAD MODEL ---
from tensorflow.keras.models import load_model

@st.cache_resource
def load_my_model():
    return load_model('models/Dense_model (2).h5', compile=False)


model = load_my_model()


# --- PREPROCESSING & GRAD-CAM FUNCTIONS ---
def preprocess_image(img, target_size=(224, 224)):
    img = np.array(img.convert('RGB'))
    # Center Crop (85%)
    h, w = img.shape[:2]
    new_h, new_w = int(h * 0.85), int(w * 0.85)
    start_x, start_y = (w - new_w) // 2, (h - new_h) // 2
    img = img[start_y:start_y + new_h, start_x:start_x + new_w]
    # Resize and Normalize
    img = cv2.resize(img, target_size)
    img_tensor = img.astype('float32') / 255.0
    return np.expand_dims(img_tensor, axis=0), img


def generate_gradcam(img_tensor, model, layer_name="conv5_block16_2_conv"):
    # Ensure model is built
    _ = model(img_tensor)

    # Extract base model (DenseNet121)
    base_model = model.layers[0]

    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(layer_name).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_tensor)
        tape.watch(conv_outputs)

        # Pass through classifier head
        x = base_output
        for layer in model.layers[1:]:
            x = layer(x)

        predictions = x
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return np.zeros(conv_outputs.shape[1:3])

    # Compute weights
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute heatmap
    cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)

    cam = tf.maximum(cam, 0)
    cam = cam / tf.reduce_max(cam) if tf.reduce_max(cam) != 0 else cam

    return cam.numpy()


# --- SIDEBAR: UPLOAD ---
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a Chest X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Process Image
    raw_img = Image.open(uploaded_file)
    img_tensor, cropped_img = preprocess_image(raw_img)

    # 2. Prediction
    prediction = model.predict(img_tensor)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # 3. Columns for Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Analysis Result")
        color = "red" if label == "PNEUMONIA" else "green"
        st.markdown(f"### Diagnosis: :{color}[{label}]")
        st.metric("Confidence", f"{confidence:.2%}")
        st.image(cropped_img, caption="Processed Image (Center Cropped)", use_container_width=True)

    with col2:
        st.subheader("Explainable AI (Grad-CAM)")
        heatmap = generate_gradcam(img_tensor, model)

        # Superimpose
        heatmap_resize = cv2.resize(heatmap, (cropped_img.shape[1], cropped_img.shape[0]))
        heatmap_resize = np.uint8(255 * heatmap_resize)
        heatmap_color = cv2.applyColorMap(heatmap_resize, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(cropped_img, 0.6, heatmap_color, 0.4, 0)

        st.image(superimposed, caption="Diagnostic Focus Area", use_container_width=True)
        st.info("The red zones indicate where the AI detected signs of lung opacity.")
else:
    st.info("Waiting for an X-ray upload...")
