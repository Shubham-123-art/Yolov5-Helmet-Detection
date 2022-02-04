# Libraries
from sklearn import preprocessing, model_selection
import os
import sys
import time
import random
from tqdm import tqdm
import warnings
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import shutil

import os
from glob import glob
import pandas as pd
import xml.etree.ElementTree as ET

OUTPUT_PATH = r"D:\Downloads\Downloads\helmet_dataset_1"
DATA_PATH = r"D:\Downloads\Downloads\Helmet dataset"

path = glob(r"D:\Downloads\Downloads\Helmet dataset\annotations\*.xml")


xml_list = []
for xml_file in path:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        # print('MEMBER', member)
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 member[0].text,
                 int(member[5][0].text),
                 int(member[5][1].text),
                 int(member[5][2].text),
                 int(member[5][3].text)
                 )
        xml_list.append(value)
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
df = pd.DataFrame(xml_list, columns=column_name)
# print(df.head())


def width(df):
    return int(df.xmax - df.xmin)


def height(df):
    return int(df.ymax - df.ymin)


df['width_1'] = df.apply(width, axis=1)
df['height_1'] = df.apply(height, axis=1)

df['X'] = df['xmin'].copy()
df['Y'] = df['ymin'].copy()
# print(df.head())

labels = df.loc[:, ['X', 'Y', 'width_1', 'height_1']].values
# print(labels)


df['bboxes'] = list(labels)
# print(df.head())

le = preprocessing.LabelEncoder()
le.fit(df['class'])
# print(le.classes_)
labels = le.transform(df['class'])
df['labels'] = labels
# print(df.tail(10))


data1 = df.groupby('filename')['bboxes'].apply(list).reset_index(name='bboxes')
# print(data1.head())

data2 = df.groupby('filename')['labels'].apply(list).reset_index(name='labels')


data3 = df.groupby('filename')['width', 'height'].agg(
    width=('width', 'max'), height=('height', 'max'))


train_df = pd.merge(data1, data2, on='filename')
train_df = pd.merge(train_df, data3, on='filename')
# print(train_df.head())


# os.path.join(r"D:\Downloads\Downloads\Helmet dataset\images", f"{train_df.filename}")
train_df['path'] = f"D:\Downloads\Downloads\Helmet dataset\images\{train_df['filename']}"
#print(train_df.loc[0:1, 'path'])


#print(train_df.loc[0, 'path'])

# train_df['path'] = DATA_PATH+ "\" + "images" + "\" + train_df['filename']


file_name = []
for i in range(len(train_df)):
    file_name.append(train_df['filename'][i].split('.')[0])


train_df['file_name'] = file_name
print(train_df.head())

# # Create sepparate paths for images and their labels (annotations)
# # these will come in handy later for the YOLO model
#train_df["path_images"] = "/kaggle/images/" + train_df['file_name'] + ".png"
#train_df["path_labels"] = "/kaggle/labels/" + train_df['file_name'] + ".txt"


# # Calculate the number of total annotations within the frame
# data["no_annotations"] = data["bboxes"].apply(lambda x: len(x))


def coco2yolo(image_height, image_width, bboxes):
    """
    Converts a coco annotation format [xmin, ymin, w, h] to
    the corresponding yolo format [xmid, ymid, w, h]

    image_height: height of the original image
    image_width: width of the original image
    bboxes: coco boxes to be converted
    return ::

    inspo: https://www.kaggle.com/awsaf49/great-barrier-reef-yolov5-train
    """

    bboxes = np.array(bboxes).astype(float)

    # Normalize xmin, w
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / image_width
    # Normalize ymin, h
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / image_height

    # Converstion (xmin, ymin) => (xmid, ymid)
    bboxes[:, [0, 1]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]] / 2

    # Clip values (between 0 and 1)
    bboxes = np.clip(bboxes, a_min=0, a_max=1)

    return bboxes


yolo_bbox = []
for i in range(len(train_df)):
    yolo_bbox.append(coco2yolo(train_df['height'][i], train_df['width'][i], train_df['bboxes'][i]))

#yolo_bbox = coco2yolo(train_df['height'], train_df['width'], train_df['bboxes'])
train_df['yolo_bbox'] = yolo_bbox
# print(train_df.head())


# df_train, df_valid = model_selection.train_test_split(
#     train_df, test_size=0.1, random_state=13, shuffle=True)
# print(df_train.shape, df_valid.shape)

def process_data(data, data_type="train"):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row['file_name']
        bounding_boxes = row['yolo_bbox']
        labels = row['labels']
        yolo_data = []
        for bbox in bounding_boxes:
            x_center = bbox[0]
            y_center = bbox[1]
            width = bbox[2]
            height = bbox[3]
            for label in labels:
                yolo_data.append([label, x_center, y_center, width, height])

        yolo_data = np.array(yolo_data)
        np.savetxt(
            os.path.join(OUTPUT_PATH, f"labels\{data_type}\{image_name}.txt"), yolo_data, fmt=["%d", "%f", "%f", "%f", "%f"]
        )
        shutil.copyfile(
            os.path.join(DATA_PATH, f"images\{image_name}.png"), os.path.join(
                OUTPUT_PATH, f"images\{data_type}\{image_name}.png")
        )


df_train, df_valid = model_selection.train_test_split(
    train_df, test_size=0.1, random_state=13, shuffle=True)
print(df_train.shape, df_valid.shape)

process_data(df_train, "train")
process_data(df_valid, "validation")

# print(process_data(train_df, "train"))

# yolo_data = []
# for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
#     image_name = row['file_name']
#     bounding_boxes = row['yolo_bbox']
#     labels = row['labels']
#     for bbox in bounding_boxes:
#         x_center = bbox[0]
#         y_center = bbox[1]
#         width = bbox[2]
#         height = bbox[3]
#         for label in labels:
#             yolo_data.append([label, x_center, y_center, width, height])

# print(yolo_data)


# # def process_data(data, data_type="train"):
#       yolo_data=[]
# #     for _, row in tqdm(data.iterrows(), total=len(data)):
# #         yolo_data = []
# #         image_name = row.filename
# image_height=row['height']
# image_width=row['width']
# labels=row['labels']
# bbox=row['bboxes']
#         yolo_data.append([row.labels, row.x_center_norm,
#                          row.y_center_norm, row.width_norm, row.height_norm])
#         yolo_data = np.array(yolo_data)
#         np.savetxt(os.path.join(OUTPUT_PATH, f"labels\{data_type}\{image_name}.txt"), yolo_data, fmt=[
#                    "%d", "%f", "%f", "%f", "%f"])

#         shutil.copyfile(os.path.join(DATA_PATH, f"images\{image_name}.png"),
#                         os.path.join(OUTPUT_PATH, f"images\{data_type}\{image_name}.png"))


# process_data(df_train, "train")
# process_data(df_valid, "validation")

# yolo_bboxes = []

# for k in tqdm(range(len(data))):

#     row_data = data.iloc[k, :]
#     height = row_data["height"]
#     width = row_data["width"]
#     coco_bbox = row_data["bboxes"]
#     len_bbox = row_data["no_annotations"]

#     # Create file and write in it
#     with open(row_data["path_labels"], 'w') as file:

#         # In case there is an image with no present annotation
#         if len_bbox == 0:
#             file.write("")
#             continue

#         # Convert coco format to yolo format
#         yolo_bbox = coco2yolo(height, width, coco_bbox)
#         yolo_bboxes.append(yolo_bbox)

#         # Write annotations in file
#         for i in range(len_bbox):
#             annot = ["0"] + \
#                     yolo_bbox[i].astype(str).tolist() + \
#                     ([""] if i+1 == len_bbox else ["\n"])

#             annot = " ".join(annot).strip()
#             file.write(annot)
