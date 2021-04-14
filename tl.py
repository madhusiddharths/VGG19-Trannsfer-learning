%%writefile app.py
from tensorflow import keras
import streamlit as st
from keras.models import load_model
from keras.applications.vgg19 import VGG19
from skimage.transform import resize
import numpy as np
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2
from keras.applications.vgg19 import VGG19,preprocess_input,decode_predictions
st.title("horse or Human Image - CLassifier")
upload = st.file_uploader('Upload the image')

#load model
model_new=keras.models.load_model('/content/model_new.h5')

# predict
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
if st.button('PREDICT'):
  model = model_new
  x = cv2.resize(opencv_image,(224,224))
  x = np.expand_dims(x,axis=0)
  x = preprocess_input(x)
  y = model.predict(x)
  if y[0][0]==1:
    st.title('The predicted output is : Horse')
  else:
    st.title('The predicted output is : Human')
