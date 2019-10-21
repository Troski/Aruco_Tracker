import argparse
import cv2
import numpy as np
import glob
import os
__author__ = "Marco Andarcia"
__copyright__ = "TBD"
__credits__ = ["Marco Andarcia"]
__license__ = "TBD"
__version__ = "1.0.0"
__maintainer__ = "Marco Andarcia"
__email__ = "marcoandarcia89@gmail.com"
__status__ = "Under Development"

class PoseEstimator:
    def __init__(self, image):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*7,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.calib_images = glob.glob('calib_images/*.jpg')
        self.gray = []

    def calibrate_camera(self):

        for image_file in self.calib_images:
            img = cv2.imread(image_file)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(self.gray, (7,6),None)
            if ret:
                self.objpoints.append(self.objp)
                corners2 = corners2 = cv2.cornerSubPix(self.gray,corners,(11,11),(-1,-1),self.criteria)
                self.imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1],None,None)
        print("Calibration Matrix: ",mtx)
        return mtx, dist, rvecs, tvecs

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    def find_pose(self, image_file):
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((6*7,3), np.float32)
            objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
            axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
                mtx, dist, rvecs, tvecs = self.calibrate_camera()
                # Find the rotation and translation vectors.
                _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img = self.draw(img,corners2,imgpts)
                cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                cv2.imshow('img',img)
                k = cv2.waitKey(0) & 0xff

        except Exception as ex:
            print("PoseEstimator: " + str(ex))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=str, default="calib_images/left01.jpg", help="Path to marker image."
    )
    args = parser.parse_args()

    if os.path.exists(args.image):
        pose_estimator = PoseEstimator(args.image)
        pose_estimator.find_pose(args.image)
    else:
        print("main: " + str(args.image) + "not found")
