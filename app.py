import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

banner_path="banner.jpg"
model=tf.keras.models.load_model("leaf_disease_coloured_24_3.h5")

menu = ["Disease Detection","About"]
choice = st.sidebar.selectbox("Select Activty", menu)
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


if choice=="About":
    intro_markdown = read_markdown_file("about.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)
elif choice=="Disease Detection":
    st.title("Grape Leaf Disease Diagnosis")
    file = st.file_uploader("Upload image...", type=["jpg", 'png'])
    def import_and_predict(image_data, model):
        size = (256, 256)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img_reshape = img_resize[np.newaxis, ...]
        print(img_reshape)
        predict_class = np.argmax(model.predict(img_reshape), axis=1)

        return predict_class


    if file is None:
        st.write("Please upload an grape leaf image")
    else:
        image = Image.open(file)
        st.sidebar.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]

        if categories[prediction[0]] == "Healthy":
            st1 = "No Disease detected"
            st.sidebar.success(st1)
        elif categories[prediction[0]] == "Black_rot":
            stringbt = "This grape is having : Blackrot"
            st.sidebar.success(stringbt)
            st.title("Treatment")
        elif categories[prediction[0]] == "Esca_(Black_Measles)":
            stringbm = "This grape is having : Esca (Black_Measles)"
            st.sidebar.success(stringbm)
            st.title("Treatment")
        elif categories[prediction[0]] == "Leaf_blight_(Isariopsis_Leaf_Spot)":
            stringlb = "This grape is having : Leaf blight (Isariopsis Leaf Spot)"
            st.sidebar.success(stringlb)
            st.title("Treatment")
