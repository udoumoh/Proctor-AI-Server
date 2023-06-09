import numpy as np
import math
import pandas as pd
import os
import glob
import base64
import cv2

def SetColumnHeading(dim=None):
    if dim == "x" or dim == "X":
        return [
            'Op0_X', 'Op1_X', 'Op2_X', 'Op3_X', 'Op4_X', 'Op5_X', 'Op6_X', 'Op7_X', 'Op8_X', 'Op9_X', 'Op10_X', 'Op11_X', 'Op12_X',
            'Op13_X', 'Op14_X', 'Op15_X', 'Op16_X', 'Op17_X', 'Op18_X', 'Op19_X', 'Op20_X', 'Op21_X', 'Op22_X', 'Op23_X', 'Op24_X'
        ]

    if dim == "y" or dim == "Y":
        return [
            'Op0_Y', 'Op1_Y', 'Op2_Y', 'Op3_Y', 'Op4_Y', 'Op5_Y', 'Op6_Y', 'Op7_Y', 'Op8_Y', 'Op9_Y', 'Op10_Y', 'Op11_Y',
            'Op12_Y', 'Op13_Y', 'Op14_Y', 'Op15_Y', 'Op16_Y', 'Op17_Y', 'Op18_Y', 'Op19_Y', 'Op20_Y', 'Op21_Y', 'Op22_Y', 'Op23_Y', 'Op24_Y'
        ]

    if dim == "z" or dim == "Z":
        return [
            'Op0_Z', 'Op1_Z', 'Op2_Z', 'Op3_Z', 'Op4_Z', 'Op5_Z', 'Op6_Z', 'Op7_Z', 'Op8_Z', 'Op9_Z', 'Op10_Z', 'Op11_Z', 'Op12_Z',
            'Op13_Z', 'Op14_Z', 'Op15_Z', 'Op16_Z', 'Op17_Z', 'Op18_Z', 'Op19_Z', 'Op20_Z', 'Op21_Z', 'Op22_Z', 'Op23_Z', 'Op24_Z'
        ]

    return [
        'Op0_X', 'Op0_Y', 'Op0_Z', 'Op1_X', 'Op1_Y', 'Op1_Z',
        'Op2_X', 'Op2_Y', 'Op2_Z', 'Op3_X', 'Op3_Y', 'Op3_Z', 'Op4_X',
        'Op4_Y', 'Op4_Z', 'Op5_X', 'Op5_Y', 'Op5_Z', 'Op6_X', 'Op6_Y',
        'Op6_Z', 'Op7_X', 'Op7_Y', 'Op7_Z', 'Op8_X', 'Op8_Y', 'Op8_Z',
        'Op9_X', 'Op9_Y', 'Op9_Z', 'Op10_X', 'Op10_Y', 'Op10_Z',
        'Op11_X', 'Op11_Y', 'Op11_Z', 'Op12_X', 'Op12_Y', 'Op12_Z',
        'Op13_X', 'Op13_Y', 'Op13_Z', 'Op14_X', 'Op14_Y', 'Op14_Z',
        'Op15_X', 'Op15_Y', 'Op15_Z', 'Op16_X', 'Op16_Y', 'Op16_Z',
        'Op17_X', 'Op17_Y', 'Op17_Z', 'Op18_X', 'Op18_Y', 'Op18_Z',
        'Op19_X', 'Op19_Y', 'Op19_Z', 'Op20_X', 'Op20_Y', 'Op20_Z',
        'Op21_X', 'Op21_Y', 'Op21_Z', 'Op22_X', 'Op22_Y', 'Op22_Z',
        'Op23_X', 'Op23_Y', 'Op23_Z', 'Op24_X', 'Op24_Y', 'Op24_Z'
    ]

def NormalizePose(pose, flipY=True):
    convertedPose = []
    # Compute the Maximum bounds in dimensions X and Y of the pose
    maxX, minX = -math.inf, math.inf
    maxY, minY = -math.inf, math.inf
    for keyPoint in pose:
        if keyPoint[2] == 0:
            continue
        if keyPoint[0] > maxX:
            maxX = keyPoint[0]
        if keyPoint[0] < minX:
            minX = keyPoint[0]
        if keyPoint[1] > maxY:
            maxY = keyPoint[1]
        if keyPoint[1] < minY:
            minY = keyPoint[1]
    frameDiffX = maxX - minX
    frameDiffY = maxY - minY
    # Convert the Coordinates to normalized values
    # NOTE: The Origin of the KeyPoints is located at the topleft of the image and is forcibly flipped around the X axis to
    # reflect the rectangular coordinate system where its logical Origin is now at the bottomleft.
    for keyPoint in pose:
        convertedKeyPoint = [0, 0, 0]
        if keyPoint[2] == 0:
            convertedPose.append(convertedKeyPoint)
            continue
        convertedKeyPoint[0] = (keyPoint[0] - minX) / (frameDiffX)
        if flipY == True:
            convertedKeyPoint[1] = (keyPoint[1] - minY) / (frameDiffY)
        else:
            convertedKeyPoint[1] = (maxY - keyPoint[1]) / (frameDiffY)
        convertedKeyPoint[2] = keyPoint[2]
        convertedPose.append(convertedKeyPoint)
    return convertedPose


def ReshapePoseCollection(poseCollection):
    numPoses, numKeyPoints, KeyPointVector = (
        poseCollection.shape[0],
        poseCollection.shape[1],
        poseCollection.shape[2],
    )
    poseCollection = np.reshape(
        poseCollection, (numPoses, numKeyPoints * KeyPointVector)
    )
    return poseCollection


def ConvertToDataFrame(poseCollection, label=None):
    columnNames = SetColumnHeading()
    poseDF = pd.DataFrame(poseCollection, columns=columnNames)
    if label is not None:
        poseDF["label"] = label
    return poseDF

def NormalizePoseCollection(poseCollection, flipY=True):
    convertedPoseCollection = []
    for pose in poseCollection:
        convertedPose = NormalizePose(pose, flipY=flipY)
        convertedPoseCollection.append(convertedPose)
    return np.array(convertedPoseCollection)

def get_image_files(folder_path):
    image_files = []
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']  # Add more extensions as needed

    # Iterate over the supported extensions and retrieve image files
    for extension in supported_extensions:
        search_pattern = os.path.join(folder_path, extension)
        image_files.extend(glob.glob(search_pattern))

    return image_files

def image_to_base64(image_data):
    _, buffer = cv2.imencode('.jpg', image_data)
    base64_data = base64.b64encode(buffer).decode('utf-8')
    return base64_data