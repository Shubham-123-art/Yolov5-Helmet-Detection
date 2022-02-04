# Yolov5-Helmet-Detection
 
 In this project i have tried to detect the helemts in videos using yolov5 object detection architecture. But before going to deep first let's talk about the dataset.
 
 Dataset Link - https://www.kaggle.com/andrewmvd/helmet-detection
 
The dataset contains 764 images of 2 distinct classes for the objective of helmet detection.
Bounding box annotations are provided in the PASCAL VOC format
The classes are:

. With helmet
. Without helmet

So, clearly we have very small number of images. And the problems increases when we are detecting helmet in second video(crazy indian traffic  https://www.youtube.com/watch?v=iEIk3RpV6RA&t=31s). This video is recorded from some height but this dataset doesn't have images of upper section of helmet so it very hard to get good accuracy in such videos until we have good quality of images of upper section of helmet.

Data preperation for Yolov5

So yolov5 wants data in certain format(Label, X_center, Y_center, Width, Height). Some information is provided in xml files in dataset (Xmin,Ymin,Xmax,Ymax,image_height,image_width). So to bring data in Yolo format I had to apply some data preprocessing technique.
