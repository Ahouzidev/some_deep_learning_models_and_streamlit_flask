import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def load_model():
    model = tf.keras.models.load_model('fruits_cnn.h5')
    return model

modelCNN = load_model()

classes=["apple","banana","orange"]
st.title("Fruits classification application")
st.write("Upload an image of a fruit to identify if it'is a **Banana** or **Orange** or **Apple**.")

uploadad_file=st.file_uploader("Upload an image here:",type=["png","jpg","jpeg"])

if uploadad_file is not None:
    image = Image.open(uploadad_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    img=image.resize((32,32))
    img_array=np.array(img)
    img_array=np.expand_dims(img_array,axis=0)

    predictions=modelCNN.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)*100

    st.subheader("**Prediction results:**")
    st.success(f"Predicted fruit: {predicted_class}")
    st.write(f"**Confidence**: {confidence}")

    st.write("class probabilities:")
    for i, label in enumerate(classes):
        st.write(f"*****{label} : {predictions[0][i]*100:.2f}%****")

else:
    st.info("Please upload an image.")
