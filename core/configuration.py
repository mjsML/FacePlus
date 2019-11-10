import numpy as np
import os
#from drivers.kinect360.RGBD
from drivers.genericWebCam import RGBcamera
from easydict import EasyDict

config = EasyDict()
config.cameraID = 0
config.defaultDriver = RGBcamera
config.defaultDetectionNetworkPath="./weights/detection/16and32"
config.defaultAlignmentNetworkPath="./weights/alignment/Alignment_Model"
config.defaultRecognitionNetworkPath="./weights/recognition/Rec_Model"

def generate_config(_camera, _detection, _alignment,_Recognition):
    pass