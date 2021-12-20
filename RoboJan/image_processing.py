import numpy as np
import cv2 as cv
import glob


def calibrate(setup, calibration):

    if calibration == False:
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((setup['chess_board_corners'][0] * setup['chess_board_corners'][1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:setup['chess_board_corners'][0], 0:setup['chess_board_corners'][1]].T.reshape(-1, 2)


        objp = objp * setup['chessboard_square_size_mm'][0]

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('C:/Users/hemza/PycharmProjects/RP3/RoboJan/callibration_images/*.jpg')

        for image in images:
            img = cv.imread(image)
            img = cv.resize(img, setup['frame_size_pixels'])
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # cv.imshow('im', gray)
            # cv.waitKey(100)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, setup['chess_board_corners'], None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            # cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(1000)

        cv.destroyAllWindows()

        ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, setup['frame_size_pixels'], None, None)
        return ret, camera_matrix, dist, rvecs, tvecs

def colour_filter(frame, blur_radius, lower_bound_colour, upper_bound_colour):
    # convert to HSV colour space (max values: 180, 255, 255)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(hsv, blur_radius, 0)
    # preparing the mask to overlay
    mask = cv.inRange(img_blur, np.asarray(lower_bound_colour), np.asarray(upper_bound_colour))
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    masked_frame = cv.bitwise_and(frame, frame, mask=mask)
    return  masked_frame

def edge_detect(masked_frame):
    # convert img to grayscale
    img_gray = cv.cvtColor(masked_frame, cv.COLOR_BGR2GRAY)
    # Canny Edge Detection
    edges = cv.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny Edge Detection
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours = np.squeeze(cv.goodFeaturesToTrack(img_gray, 4, 0.1, 10))
    return edges, contours

def display_contours(contours):
    cv.drawContours(frame_processed, contours, -1, (0, 255, 0))

def area_filter(contours):
    max_area = -1
    coordinates = contours
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > max_area:
            coordinates = contours[i]
            max_area = area
    coordinates = cv.approxPolyDP(coordinates, 0.01 * cv.arcLength(coordinates, True), True)
    return coordinates

def coordinate_filter(coordinates, coordinates_filtered, coordinates_old):
    if coordinates.size == 8:
        coordinates_filtered = np.squeeze(coordinates) # turns annoying 3d even tho z axis is depth 1 to a nice 2d

    for i in range(len(coordinates_filtered)):
        distance_coordinate_0_1 = np.square((coordinates_filtered[0][:] - coordinates_filtered[1][:]))
        distance_coordinate_0_1 = np.sqrt(distance_coordinate_0_1[0] + distance_coordinate_0_1[1])
        distance_coordinate_1_2 = np.square((coordinates_filtered[1][:] - coordinates_filtered[2][:]))
        distance_coordinate_1_2 = np.sqrt(distance_coordinate_1_2[0] + distance_coordinate_1_2[1])

    if distance_coordinate_1_2 > distance_coordinate_0_1:
        coordinates_filtered = np.roll(coordinates_filtered, 1, axis=0)
        # print("rolled")
    if (np.average(coordinates_old[0] - coordinates_filtered[0])) > np.average((coordinates_old[0] - coordinates_filtered[1])): # really shit filter that checks that dont jump by 180 deg between filters. rewrite this pls
        coordinates_filtered = np.roll(coordinates_filtered, 2, axis=0)

    return coordinates_filtered

def display_tracker(coordinates_filtered, rvecs, tvecs, camera_matrix, dist, frame):
    end_point2D, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 1.0)]), rvecs, tvecs, camera_matrix, dist)

    point1 = (int(coordinates_filtered[0][0]), int(coordinates_filtered[0][1]))

    point2 = (int(end_point2D[0][0][0]), int(end_point2D[0][0][1]))

    cv.line(frame, point1, point2, (0, 0, 255), 2)

    return frame

