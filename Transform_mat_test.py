import cv2 as cv
import numpy as np
from operator import itemgetter
from glob import glob
import matplotlib.pyplot as plt
from numpy.linalg import inv



camera_matrix = np.array([442.282, 0, 315.189, 0, 444.192, 252.185, 0, 0, 1]).reshape(3,3)
old_frame = np.array([150, 132, 153, 239, 473, 235, 473, 130]).reshape(4,2)
new_frame = np.array([151, 133, 155, 241, 473, 237, 472, 131]).reshape(4,2)

objectpoints = np.array(0, 0, 0, 3, 0, 0, 3, 1, 0, 0, 1, 0).reshape(4,3)

# m1 = np.zeros([8,8])
#
# for i in range(np.shape(old_frame)[0]):
#     m1[(i*2)] = [old_frame[i,0], old_frame[i,1], 1, 0, 0, 0, -old_frame[i,0]*new_frame[i,0], -old_frame[i,1]*new_frame[i,0]]
#     m1[(i*2)+1] = [0, 0, 0, old_frame[i,0], old_frame[i,1], 1, -old_frame[i,0]*new_frame[i,1], -old_frame[i,1]*new_frame[i,1]]
#
#     #{xA, yA, 1, 0, 0, 0, -xA*xB, -yA*xB}
#     #{0, 0, 0, xA, yA, 1, -xA * yB, -yA * yB}
# #https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
#
# m2 = new_frame.reshape(8,1)
#
# homography_matrix = (inv((np.transpose(m1)*m1))) * (np.transpose(m1)*m2) # magic
#
# print(homography_matrix)
# #rot, trans, norm = cv.decomposeHomographyMat(homography_matrix, camera_matrix)

H = cv.findHomography(old_frame, new_frame)

solutions = cv.decomposeHomographyMat(H[0], camera_matrix)

# #rectify coordinates into 3d
# old_frame = np.append(old_frame, [[1], [1], [1], [1]], axis=1)
# new_frame = np.append(new_frame, [[1], [1], [1], [1]], axis=1)
# print(old_frame)

#split weird data structure into variables
number_of_solutions = solutions[0]
rotation = np.array(solutions[1])
translation = np.array(solutions[2])
normals = solutions[3]

print(translation)

#camera global coordinates
camera_pos = np.array([0, 0, 500, 270, 0, 0])



#print(np.linalg.norm(normals[3]))

#doesnt work!!!
#solutions = cv.filterHomographyDecompByVisibleRefpoints(rotation, translation, old_frame, new_frame)


#
# 1. Sort solutions into rotations, translations and normals - done
# 2. Run through Homographydecompbyvisible to go from 4 solutions to 2 - wtff what are rectified coordinates
# 3. work out which one of the 2 is correct
# 4. apply to all frames and store data
# 5. plot movement on graph