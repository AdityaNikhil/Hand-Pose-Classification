# Hand-Pose-Classification

This repo is based on classifying various hand poses such as, 👍👎☝👌👊👉👈✋ using a hand pose estimation model in the back end.<br>
The poses are trained using a hand pose model whose keypoints are then extracted and fed into an MLP classifier to classify various hand poses using the keypoints as inputs. <br>
Hand pose model : https://www.kaggle.com/changethetuneman/openpose-model

## Architecture of the model
<img src="https://github.com/AdityaNikhil/Hand-Pose-Classification/blob/master/assets/Model_arch.jpg" width=800 /> ![]()

As you can see from the above img,<br>
    1) The pre-trained caffe model takes a hand image as input. <br>
    2) Then extracts all the keypoints(22) from the image. (There are 22 X and 22 Y keypoints)<br>
    3) Now these keypoints are fed into an neural network for classification.<br>
    4) Then the neural net classifies the image based on keypoints.
    
## Why classify based on keypoints rather than image?!
It's always been so easy to classify images using neural nets or any other classification algorithm. The main reason for classifying with keypoints here is that,<br>
**The neural net generally learns from a set of images it was trained while it gets confused when a new img is given.<br> 
While here, it's simply learning the various keypoints in images and it'll be easily able to classify any img based on these keypoints**. <br> Hence,
we're able to save a lot of time for training and inference is carried out easily.

## Running the files
Below, run only the files which are asked to. To see the results.<br>

1) **handPoseImage.py** -- Run this file to see how the pre-trained caffe model does inference on img. <br>
2) **handPoseVideo.py** -- Run this file to see how the pre-trained caffe model does inference on videos. <br>
3) **handPoseCamera.py** -- Run this file to see how the pre-trained caffe model does inference using camera in real time. <br>
4) **Extract_KPts.ipynb** -- Extracts the keypoints from input image.<br>
5) **Train_MLP_Classifier.ipynb** -- Training a neural net to classify input imgs based on keypoints. <br>
6) **inference.py** -- Run this file to see the inference on a given img or video for classification.<br>
        For images,<br>
            `python inference.py -i image.jpg` <br>
        For videos,<br>
            `python inference.py -i demo.mp4` <br>         
`**NOTE** - Currently this repo is trained only on 👍 👎 hand poses. More poses and user interactions will be added soon to make it more fun. `

## TODOs
        1) Make a cmd argument parser to parse the following arguments,
            1.1) model path
            1.2) video or img or real-time
            1.3) Default :- protoFile, WeightsFile, model(Until more models are created)
        2) Train on more poses.
        3) Make an interface using streamlit. (checkout below image)
        4) Real-time training in streamlit just like teacchable machine using cloud GPUs.
        
<img src="https://github.com/AdityaNikhil/Hand-Pose-Classification/blob/master/assets/streamlit_inference.jpg" width=800 /> ![]()      
