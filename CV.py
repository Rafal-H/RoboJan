import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import Calibrate_Camera

ret, cameraMatrix, dist, rvecs, tvecs = Calibrate_Camera.calibrate()

objectpoints = np.array([0, 0, 0, 3, 0, 0, 3, 1, 0, 0, 1, 0], np.float32).reshape(4,3) #3d points in arbitrary coordinates


#camera = cv.VideoCapture('360ptest.mp4') # Specify file
camera = cv.VideoCapture('Test2_50cmstart.mp4') # live video feed from a camera (camera0)

# empty canvas
black = np.zeros((360, 640, 3), np.uint8) #360p black image by making array of 0s

#original points that get tracked in frame 1
originView = np.array([150, 132, 153, 239, 473, 235, 473, 130]).reshape(4,2) # starts bottom right goes counterclockwise (bruh is that right?!)

#camera starting position
x = [p[0] for p in originView]
y = [p[1] for p in originView]
centroid = (int(sum(x) / len(originView)), int(sum(y) / len(originView)))

#camera global coordinates
camera_pos = np.array([0, 0, 500, 270, 0, 0]) # x, y, z, pitch, yaw, roll

#runway global coordinates
runway_pos = originView - centroid
runway_pos = np.hstack((runway_pos, np.zeros((4,1))))  # add z positions in 3rd column as 0s

#colour filter
lower_bound = np.array([100, 40, 100])  # set min man max colour space values
upper_bound = np.array([190, 255, 255])


#main loop frame by frame
while True:
    _, frame = camera.read()

    cv.imshow('Unprocessed', frame)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) #convert to HSV colour space (max values: 180, 255, 255)

    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(hsv, (5, 5), 0)
    # preparing the mask to overlay
    mask = cv.inRange(img_blur, lower_bound, upper_bound)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    masked = cv.bitwise_and(frame, frame, mask=mask)

    # Convert to graycsale
    img_gray = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)

    # Canny Edge Detection
    edges = cv.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny Edge Detection

    # Display Canny Edge Detection Image
    frame_processed = frame
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame_processed, contours, -1, (0, 255, 0))
    #print(contours)

    max_area = -1
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > max_area:
            coordinates = contours[i]
            max_area = area
    coordinates = cv.approxPolyDP(coordinates, 0.01 * cv.arcLength(coordinates, True), True)
    #print(coordinates)
    # coordinates_filtered = coordinates
    # if finds more than 4 verticies then use last 4 from previous frame
    if coordinates.size == 8:
        coordinates_filtered = np.squeeze(coordinates) # turns annoying 3d even tho z axis is depth 1 to a nice 2d
        print(coordinates_filtered)


    for i in range(len(coordinates_filtered)):
        dots = cv.circle(frame_processed, (coordinates_filtered[i, 0], coordinates_filtered[i, 1]), radius=3, color=(0, 0, 255), thickness=-1)
        # dots = cv.circle(frame_processed, (originView[i, 0], originView[i, 1]), radius=3,color=(250, 60, 100), thickness=2)

    #dots = cv.circle(frame_processed, (int(centroid[0]), int(centroid[1])), radius=5, color=(250, 60, 100),thickness=5)

    distance_coordinate_0_1 = np.square((coordinates_filtered[0][:] - coordinates_filtered[1][:]))
    distance_coordinate_0_1 = np.sqrt(distance_coordinate_0_1[0] + distance_coordinate_0_1[1])

    distance_coordinate_1_2 = np.square((coordinates_filtered[1][:] - coordinates_filtered[2][:]))
    distance_coordinate_1_2 = np.sqrt(distance_coordinate_1_2[0] + distance_coordinate_1_2[1])

    if distance_coordinate_1_2 > distance_coordinate_0_1:
        coordinates_filtered = np.roll(coordinates_filtered, 1, axis=0)
        print(coordinates_filtered)
        print("rolled")

    ret, rvecs, tvecs = cv.solvePnP(objectpoints, coordinates_filtered.astype(np.float32), cameraMatrix, dist)
    nose_end_point2D, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 1.0)]), rvecs, tvecs, cameraMatrix, dist)

    point1 = (int(coordinates_filtered[0][0]), int(coordinates_filtered[0][1]))

    point2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv.line(frame, point1, point2, (255, 255, 255), 2)




    black = np.zeros((360, 640, 3), np.uint8)  # 360p black image by making array of 0s
    outline = cv.polylines(black, [coordinates_filtered], True, (255,255,0), 2)

    #print(originView)
    #print(coordinates_filtered)

    # doesnt work cause not what I want
    #transform_mat = cv.getPerspectiveTransform(originView.astype(np.single), coordinates_filtered.astype(np.single))

    #print(transform_mat)

    cv.imshow('coordinates', frame_processed)
    cv.imshow('Canny Edge Detection', edges)
    cv.imshow("Mask", masked)
    cv.imshow('quadrilateral', outline)


    # #3D Plotting
    # plt.style.use('fivethirtyeight')
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(runway_pos[:, 0], runway_pos[:, 1], runway_pos[:, 2])
    # ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2])
    # plt.xlim([-500, 500])
    # plt.ylim([-500, 500])
    # plt.show()

    if cv.waitKey(10) == ord("x"):
        break

camera.release()
cv.destroyAllWindows()

# planar projection is called homography. Check email he sent with paper and solve this.
# More interesting project is to run a decision tree with 3 sources of information: vision + gps + expected position. When one goes wrong use other 2 to not shit urself
