import cv2 as cv
import Calibrate_Camera
import numpy as np


ret, cameraMatrix, dist, rvecs, tvecs = Calibrate_Camera.calibrate()

# objectpoints = np.array([0, 0, 0, 3, 0, 0, 3, 1, 0, 0, 1, 0], np.float32).reshape(4,3)
objectpoints = np.array([0, 0, 0, 3, 0, 0, 3, 1, 0, 0, 1, 0], np.float32).reshape(4,3)
print(objectpoints)

old_frame = np.array([150, 132, 153, 239, 473, 235, 473, 130]).reshape(4,2)
new_frame = np.array([151, 133, 155, 241, 473, 237, 472, 131], np.float32).reshape(4,2)

ret, rvecs, tvecs = cv.solvePnP(objectpoints, new_frame, cameraMatrix, dist)