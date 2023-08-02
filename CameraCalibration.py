import numpy as np
import cv2 as cv
import glob


class CameraCalibration:

    def __init__(self, m, n):
        self.__m = m
        self.__n = n

        self.__CalibrateCamera(self.__m, self.__n)

    def __CalibrateCamera(self, m, n):

        try:
            # Termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # points (0,0,0), (1,0,0),...
            objp = np.zeros((n * m, 3), np.float32)
            objp[:, :2] = np.mgrid[0:m, 0:n].T.reshape(-1, 2)

            # Arrays to store object points and image points from all the images.
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.
            images = glob.glob('Images/*.jpg')
            for fname in images:
                img = cv.imread(fname)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, (m, n), None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)
                    
                    # Draw and display the corners
                    cv.drawChessboardCorners(img, (m, n), corners2, ret)

            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            np.savetxt('Files/CameraMatrix.txt', mtx)
            np.savetxt('Files/DistortionCoefficients.txt', dist)
            print("Camera and distortion coefficients saved to files.")

        except Exception as e:
            print("Error detected: ", e)
