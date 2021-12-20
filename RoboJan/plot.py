import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

with open('data.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

data = np.zeros([len(lines), 3])

for i in range(len(lines)):
    data[i,:] = lines[i].split()

file.close()


ax = plt.axes(projection='3d')

ax.scatter3D(data[:,0], data[:,1], data[:,2], cmap='Greens')
plt.show()