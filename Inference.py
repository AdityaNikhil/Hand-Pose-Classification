import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import nbimporter 
from Extract_Kpts import extract_kpts
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
ap.add_argument("-w", "--weights", type=str,
	default=r"D:\Project_Ideas\TeachableMachine_Local\handPose\hand\pose_iter_102000.caffemodel",
	help="path to weights directory")
ap.add_argument("-p", "--proto", type=str,
	default=r"D:\Project_Ideas\TeachableMachine_Local\handPose\hand\pose_deploy.prototxt",
	help="path to proto file directory")
ap.add_argument("-m", "--model", type=str,
	default=r"D:\Project_Ideas\TeachableMachine_Local\handPose\hand\Kpts_model.h5",
	help="path to trained hand classifier model")


args = vars(ap.parse_args())
model_1 = load_model(args['model'])
protoFile = args['proto']
weightsFile = args['weights']
path = args['input']

if(path.endswith('.jpg') or path.endswith('.png') or path.endswith('jpeg')):

	img = cv2.imread(path)

	points = extract_kpts.inference_img(protoFile,weightsFile,path)



	points = pd.DataFrame(points, columns=['X','Y'])
	df = pd.DataFrame(points)
	a = df[['X','Y']].to_numpy().reshape(-1, 44)
	       
	df1 = pd.DataFrame(a)
	label = 'The predicted value is : {}'.format(np.argmax(model_1.predict(df1),axis=1)[0])

	img = cv2.putText(img, label, (0,30),
			cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3)

	cv2.imshow('image',img)
	cv2.waitKey(0)


elif(path.endswith('.mp4')):	

	extract_kpts.inference_vid(path,protoFile,weightsFile,path,model_1)
