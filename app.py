import streamlit as st
import tensorflow as tf


# st.set_option('deprection.showFileUploaderEncoding',False)
st.cache(allow_output_mutation=True)


def load_model():
    new_model=tf.keras.models.load_model("leaf_disease_coloured.h5")
    return new_model
model=load_model()
st.title("Grape Leaf Disease Detection")

file=st.file_uploader("Please upload grape leaf image",type=["jpg",'png'])
import cv2
from PIL import Image,ImageOps
import numpy as np

def import_and_predict(image_data,model):
    size=(256,256)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    predict_class=np.argmax(model.predict(img_reshape),axis=1)
    
    return predict_class

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    categories=["Black_rot","Esca_(Black_Measles)","Healthy","Leaf_blight_(Isariopsis_Leaf_Spot)"]

    if categories[prediction[0]]=="Healthy":
        s1="No Disease Detected"
        st.success(s1)
    else:
        string="This grape is having : "+categories[prediction[0]]
        st.success(string)
