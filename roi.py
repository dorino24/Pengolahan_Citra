import cv2
import numpy as np

image = cv2.imread('1.png')

roi_points = [(550, 850), (850, 350), (600, 250),(150, 650) ]
roi_points1 = [(850, 350), (900, 200), (730, 150),(600, 250) ]
roi_points2 = [(900, 200), (900, 50), (900, 10),(730, 150) ]

roi_points = np.array(roi_points, np.int32)
roi_points1 = np.array(roi_points1, np.int32)
roi_points2 = np.array(roi_points2, np.int32)

cv2.polylines(image, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)
cv2.polylines(image, [roi_points1], isClosed=True, color=(255, 0, 255), thickness=2)
cv2.polylines(image, [roi_points2], isClosed=True, color=(0, 255, 255), thickness=2)

cv2.imshow('Gambar dengan ROI', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
