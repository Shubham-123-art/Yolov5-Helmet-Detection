# Yolov5-Helmet-Detection
 
 In this project i have tried to detect the helemts in videos using yolov5 object detection architecture. But before going too deep into the project first let's talk about the dataset.
 
 Dataset Link - https://www.kaggle.com/andrewmvd/helmet-detection
 
The dataset contains 764 images of 2 distinct classes for the objective of helmet detection.
Bounding box annotations are provided in the PASCAL VOC format
The classes are:

. With helmet
. Without helmet

So, clearly we have very small number of images. And the problems increases when we are detecting helmet in second video(crazy indian traffic  https://www.youtube.com/watch?v=iEIk3RpV6RA&t=31s). This video is recorded from some height but this dataset doesn't have images of upper section of helmet so it very hard to get good accuracy in such videos until we have good quality of images of upper section of helmet.

Data preperation for Yolov5

So yolov5 wants data in certain format(Label, X_center, Y_center, Width, Height). Some information is provided in xml files in dataset (Xmin,Ymin,Xmax,Ymax,image_height,image_width). So to bring data in Yolo format I had to apply some data preprocessing technique.

Once we have converted data into yolo format After that we just had to pass this data to the yolo architecture and trained the object detection model.

After that we have used the opencv to detect helmet in the videos. We have used pafy to create a video streaming object to extract video frame by frame to make prediction on it.

We have load Yolov5 model from pytorch hub and passed our newly trained weights. It takes a single frame as input, and scores the frame using yolov5 model and returns labels and coordinates of objects detected by model in the frame.We used these labels and coordinates predicted by model on the given frame and create bounding boxes and labels.

In the end we have wrote the output in the new file.
