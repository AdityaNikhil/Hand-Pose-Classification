import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import nbimporter 
from Utils.Extract_Kpts import extract_kpts
import matplotlib.pyplot as plt
import os
from PIL import Image


model_1 = load_model("./hand/Kpts_model.h5")
protoFile = "./hand/pose_deploy.prototxt"
weightsFile = "./hand/pose_iter_102000.caffemodel"

'''
# Hand Pose Classification

### Try the app by uploading an image file from the left.
'''
st.write('1) Upload an image of 👍 or 👎.')
st.write('2) The image will be processed and output will be displayed below. ')
st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.header('User Input Features')

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=None)
if uploaded_file is not None:

	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded Image.', use_column_width=False, width=500)

	st.write("### Working on it.....")

	img = plt.imread(uploaded_file)

	points = extract_kpts.inference_img(protoFile,weightsFile,uploaded_file)

	points = pd.DataFrame(points, columns=['X','Y'])
	df = pd.DataFrame(points)
	a = df[['X','Y']].to_numpy().reshape(-1, 44)
	       
	df1 = pd.DataFrame(a)

	output = np.argmax(model_1.predict(df1),axis=1)[0]

	if output == 1:
		label = "## Thumbs Up!"
	else:
		label = "## Thumbs down!"

	st.write(label)






