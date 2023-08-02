import numpy as np
import cv2
import os
from cv2 import aruco


class PoseEstimation:

    def RunPoseEstimation(self):

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        # parameters = aruco.DetectorParameters_create()
        ids_to_find = [50, 58]

        # Importing camera parameters
        calib_path = "Files/"
        intrinsic_params_mtx = np.loadtxt(calib_path + "CameraMatrix.txt", usecols=range(3))
        dist_coefficients = np.loadtxt(calib_path + "DistortionCoefficients.txt", usecols=range(5))

        # Load Obj File
        obj = self.GetObjFromFile("Files", "Cube.txt")

        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture('rtsp://root:root@10.100.10.105/axis-media/media.amp?videocodec=h264&resolution=640x480')

        if not cap.isOpened():
            print("Exiting...")
            exit()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Exiting...")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers( image=gray, dictionary=aruco_dict, cameraMatrix=intrinsic_params_mtx, distCoeff=dist_coefficients)

            if ids is not None:
                # Create numpy array based on frame
                # Z Buffer
                z_buffer = np.full(frame.shape[::-1][1:], 99999, dtype=None, order='C')
                imgpts = None
                index = -1
                for id in ids:
                    if id == ids_to_find[0]:  # Marker 50
                        index = index + 1

                        marker_verts_cube = obj

                        ret = self.__EstimatePoseForArucoMarker(corners[index], intrinsic_params_mtx, dist_coefficients)

                        rotation_matrix = self.__CalculateRotationMatrixWithRodrigues(ret)

                        ZPoints = self.__CalculateZPointsInWorld(rotation_matrix, translation_vector=ret[1][0, 0, :], marker_verts=marker_verts_cube)

                        imgpts = self.__ProjectPointsIntoImagePlane(marker_verts=marker_verts_cube, rotation_angles=ret[0][0, 0, :], 
                        translation_vector=ret[1][0, 0, :], intrinsic_params_mtx=intrinsic_params_mtx, dist_coefficients=dist_coefficients)

                        frame, z_buffer = self.__FillZBuffer(z_buffer, frame, objColor=(0, 0, 255), ZPoints=ZPoints, imgpts=imgpts)

                    if id == ids_to_find[1]:  # Marker 58
                        index = index + 1
                        marker_verts_cube2 = obj

                        ret = self.__EstimatePoseForArucoMarker(corners[index], intrinsic_params_mtx, dist_coefficients)

                        rotation_matrix = self.__CalculateRotationMatrixWithRodrigues(ret)

                        ZPoints = self.__CalculateZPointsInWorld(rotation_matrix, translation_vector=ret[1][0, 0, :], marker_verts=marker_verts_cube2)

                        imgpts = self.__ProjectPointsIntoImagePlane(marker_verts=marker_verts_cube2, rotation_angles=ret[0][0, 0, :], translation_vector=ret[1][0, 0, :], 
                        intrinsic_params_mtx=intrinsic_params_mtx, dist_coefficients=dist_coefficients)

                        frame, z_buffer = self.__FillZBuffer(z_buffer, frame, objColor=(255, 0, 0), ZPoints=ZPoints, imgpts=imgpts)


                #cv2.imwrite("CapturedFrame.jpg", frame)

            cv2.imshow("AR frame", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        

    def GetObjFromFile(self, directory, filename):
        fname = os.path.join(directory, filename)
        if os.path.exists(fname):
            return np.genfromtxt(fname, dtype='float', delimiter=',')
        return None

    def __EstimatePoseForArucoMarker(self, corners, intrinsic_params_mtx, dist_coefficients):

        marker_size = 10  # in cm's
        return aruco.estimatePoseSingleMarkers(corners, marker_size, intrinsic_params_mtx, dist_coefficients)

        
    def __CalculateRotationMatrixWithRodrigues(self, ret):
        rotation_angles = ret[0][0, 0, :]

        # Rotation matrix with Rodrigues
        return cv2.Rodrigues(rotation_angles)


    def __FillZBuffer(self, z_buffer, frame, objColor, ZPoints, imgpts):
        index = -1
        for x, y in imgpts:
            index = index + 1
            # Check if point 3d point isnt projected outside of the positive frame bounderies
            if x < frame.shape[0] and y < frame.shape[1]:
                if ZPoints[index] < z_buffer[y, x]:
                    z_buffer[y, x] = ZPoints[index]
                    # frame[y, x] = objColor
                    frame = cv2.circle(frame, (x, y), 5, objColor, -1)

        return (frame, z_buffer)

    def __CalculateZPointsInWorld(self, rotation_matrix, translation_vector, marker_verts):

        r_t = np.append(rotation_matrix[0], np.vstack(translation_vector), axis=1)
        aux = np.ones((marker_verts.shape[0], 1))
        marker_verts_homogenous = np.concatenate((marker_verts, aux), axis=1)
        ZPoints = np.matmul(r_t, marker_verts_homogenous.T)

        return ZPoints[2]

    # Returns pixels corresponding to the model points
    def __ProjectPointsIntoImagePlane(self, marker_verts, rotation_angles, translation_vector, intrinsic_params_mtx, dist_coefficients):
        imgpts = cv2.projectPoints(marker_verts, rotation_angles, translation_vector,
                                   cameraMatrix=intrinsic_params_mtx, distCoeffs=dist_coefficients)

        imgpts = np.int32(imgpts[0]).reshape(-1, 2)
        imgpts = imgpts[(imgpts > 0).all(1)]

        return imgpts
