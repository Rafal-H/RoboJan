import image_processing
import initialise
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

coordinates_filtered = np.zeros([4,2])

setup, calibration = initialise.get_parameters('C:/Users/hemza/PycharmProjects/RP3/RoboJan/parameters.config')
object_points = initialise.generate_object_points(setup)

# fig = plt.figure()
# ax = plt.axes(projection='3d')

coordinates_old = np.zeros([4,2])


if calibration == False:
    ret, camera_matrix, dist, rvecs, tvecs = image_processing.calibrate(setup, calibration)

camera = cv.VideoCapture('C:/Users/hemza/PycharmProjects/RP3/Test2_50cmstart.mp4')

file_object = open('data.txt', 'w')

while True:
    _, frame = camera.read()
    masked_image = image_processing.colour_filter(frame, setup['blur_radius'], setup['lower_bound_colour_filter'], setup['upper_bound_colour_filter'])
    edges, contours = image_processing.edge_detect(masked_image)
    coordinates = image_processing.area_filter(contours)
    coordinates_filtered = image_processing.coordinate_filter(coordinates, coordinates_filtered, coordinates_old)
    # coordinates_filtered = contours
    ret, rvecs, tvecs = cv.solvePnP(object_points, coordinates_filtered.astype(np.float32), camera_matrix, dist)

    # STUFF THAT NEEDS TO GO INTO FUNCTIONS
    black = np.zeros((360, 640, 3), np.uint8)  # 360p black image by making array of 0s
    outline = cv.polylines(black, np.int32([coordinates_filtered]), True, (255, 255, 0), 2)
    outline = image_processing.display_tracker(coordinates_filtered, rvecs, tvecs, camera_matrix, dist, outline)
    rotation_matrix, jacobian = cv.Rodrigues(np.matrix(rvecs))
    camera_pose = -np.matrix(rotation_matrix).T * np.matrix(tvecs)

    file_object.write(str(camera_pose[0])[2:-2])
    file_object.write((' ' + (str(camera_pose[1])[2:-2])))
    file_object.write((' ' + (str(camera_pose[2])[2:-2])))
    file_object.write('\n')

    cv.imshow('Tracked', outline)
    cv.imshow('filtered', masked_image)
    cv.imshow('Original', frame)
    cv.imshow('Canny', edges)
    coordinates_old = coordinates_filtered

    # ax.scatter3D(camera_pose[0], camera_pose[1], camera_pose[2], cmap='Greens')
    # plt.pause(0.1)

    if cv.waitKey(10) == ord("x"):
        break

# plt.show()
file_object.close()
camera.release()
cv.destroyAllWindows()