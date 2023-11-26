import cv2
import numpy as np
import os
import tqdm
import regex as re

import os
path = 'D:\\Downloads\\GH061368'
label_path = 'D:\\Downloads\\exp'

maps = "D:\\Downloads\\GH061368_streetlights_map_points"

map_files = [f for f in os.listdir(maps) if os.path.isfile(os.path.join(maps,f))]

dict_maps = {}

for f in map_files:
    dict_maps[int(os.path.splitext(f)[0].split(".")[0].split("_")[-1])] = f

print(dict_maps)


labels = os.listdir(os.path.join(label_path,'labels'))
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
print(len(files))
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter('video.avi', fourcc, 30, (1440, 1920))


files.sort(key=lambda f: int(re.sub('\D', '', f)))


def findClosest(arr, n, target):
 
    # Corner cases
    if (target <= arr[0]):
        return arr[0]
    if (target >= arr[n - 1]):
        return arr[n - 1]
 
    # Doing binary search
    i = 0; j = n; mid = 0
    while (i < j):
        mid = (i + j) // 2
 
        if (arr[mid] == target):
            return arr[mid]
 
        # If target is less than array
        # element, then search in left
        if (target < arr[mid]) :
 
            # If target is greater than previous
            # to mid, return closest of two
            if (mid > 0 and target > arr[mid - 1]):
                return getClosest(arr[mid - 1], arr[mid], target)
 
            # Repeat for left half
            j = mid
         
        # If target is greater than mid
        else :
            if (mid < n - 1 and target < arr[mid + 1]):
                return getClosest(arr[mid], arr[mid + 1], target)
                 
            # update i
            i = mid + 1
         
    # Only single element left after search
    return arr[mid]
 
 
# Method to compare which one is the more close.
# We find the closest by taking the difference
# between the target and both values. It assumes
# that val2 is greater than val1 and target lies
# between these two.
def getClosest(val1, val2, target):
 
    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1
 
# Driver code
arr = list(dict_maps.keys())
arr.sort()
n = len(arr)

print(arr)

repeat  = 3

for j in tqdm.tqdm(range(len(files))):
    img = cv2.imread(os.path.join(path, files[j]))
    if os.path.splitext(files[j])[0]+".txt" in labels:
      ind = int(os.path.splitext(files[j])[0])
      close = findClosest(arr, n, ind)

      img1 = cv2.imread(os.path.join(maps, dict_maps[close]))
      img = cv2.resize(img, (1440, 960), interpolation = cv2.INTER_AREA)
      img1 = cv2.resize(img1, (1440, 960), interpolation = cv2.INTER_AREA)
      img = np.concatenate((img, img1), axis=0)
      for i in range(repeat):
        video.write(img)
    else:
        img = cv2.resize(img, (1440, 1920), interpolation = cv2.INTER_AREA)
        video.write(img)

cv2.destroyAllWindows()
video.release()
