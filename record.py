import time
import pyautogui
import numpy as np

points = list()
for i in range(100):
    pos = pyautogui.position()
    points.append([pos.x, pos.y])

    print(i)
    time.sleep(.1)

arr = np.array(points)

np.savetxt("data.csv", arr, delimiter=',')