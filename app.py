from flask import Flask, request
import base64
import cv2
import xgboost as xgb
import copy
import sys
import os
import glob
from utils import *
from flask_cors import CORS

dir_path = 'C:/Users/JOHN/Documents/openpose/build'
sys.path.append(dir_path + '/python/openpose/Release')
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' +  dir_path + '/bin;'

import pyopenpose as op

# Set OpenPose parameters
params = dict()
params["model_folder"] = "C:/Users/JOHN/Documents/openpose/models"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"

xgboost_model_path = "C:/Users/JOHN/Desktop/gembacud/Cheating-Detection/CheatDetection/XGB_BiCD_Tuned_GPU_05.model"
model = xgb.XGBClassifier()
model.load_model(xgboost_model_path)
model.set_params(**{"predictor": "gpu_predictor"})


# Create OpenPose instance
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()

app = Flask(__name__)

# Increase the maximum request body size
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Increase the maximum size of request headers
app.config['MAX_HTTP_HEADER_SIZE'] = 8192  # 4KB

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/upload", methods=["POST"])
def handle_file_post():
    data = request.get_json()
    video_url_base64 = data.get("videoLink")
    video_data = base64.urlsafe_b64decode(video_url_base64[22:])

    file_path = "C:/Users/JOHN/Videos/video.mp4"
    with open(file_path, "wb") as file:
        file.write(video_data)

    cap = cv2.VideoCapture(file_path)
    frame_counter = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    cheatingPoses = []
    def DetectCheat(Keypoints, ShowPose=True, img=None):
        poseCollection = Keypoints
        detectedPoses = []
        cheating = False
        if ShowPose == True:
            OutputImage = datum.cvOutputData
        else:
            OutputImage = image
        if poseCollection.ndim != 0:
            original_posecollection = copy.deepcopy(poseCollection)
            poseCollection = NormalizePoseCollection(poseCollection)
            poseCollection = ReshapePoseCollection(poseCollection)
            poseCollection = ConvertToDataFrame(poseCollection)
            preds = model.predict(poseCollection)
            for idx, pred in enumerate(preds):
                if pred:
                    cheating = True
        return cheating

    while True:
        ret, image = cap.read()
        if not ret:
            break
 
        frame_counter += 1
        if frame_counter % 10 == 0:  # process every 10th frame
            # Process the frame with OpenPose
            datum = op.Datum()
            datum.cvInputData = image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Get pose keypoints and output image
            keypoints = datum.poseKeypoints
            output_image = datum.cvOutputData

            if DetectCheat(keypoints):
                # Save the output_image
                # cv2.imwrite(f'C:/Users/JOHN/Desktop/cheating-detection/client/public/images/detectedImages/cheating_frame_{frame_counter}.jpg', output_image)
                cheatingPoses.append(get_image_files(output_image))
        if frame_counter >= total_frames:  # Break out of the loop when all frames have been processed
            print(cheatingPoses[0])
            break


if __name__ == "__main__":
    app.run(debug=True)
