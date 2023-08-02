import numpy as np
import cv2 as cv
import glob
from CameraCalibration import CameraCalibration
from pose_estimation import PoseEstimation


if __name__ == '__main__':
    print("Starting program...")
    m = 9
    n = 6
    CameraCalibration(m, n)

    try:
        print("Running Estimation...")
        posEst = PoseEstimation()
        posEst.RunPoseEstimation()
        print(0)
    except Exception as e:
        print("Exception occured: ", e)
