import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('cctv1.mp4')  # Ganti dengan path ke video Anda
# Tentukan titik-titik sudut ROI
roi_points = [(1200, 700), (1050, 450), (880, 450),(900, 700) ]
roi_points1 = [(1050, 450), (1000, 330), (870, 330),(880, 450) ]
roi_points2 = [(1000, 330), (920, 200), (850, 200),(870, 330) ]

# Ubah titik-titik sudut menjadi array NumPy
roi_points = np.array(roi_points, np.int32)
roi_points1 = np.array(roi_points1, np.int32)
roi_points2 = np.array(roi_points2, np.int32)

# Tentukan titik-titik sudut ROI


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Menggambar ROI di atas setiap frame video
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255,0 ), thickness=2)
    cv2.polylines(frame, [roi_points1], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.polylines(frame, [roi_points2], isClosed=True, color=(0, 0, 255), thickness=2)

    # Tampilkan frame dengan ROI
    cv2.imshow('Frame dengan ROI', frame)

    if cv2.waitKey(25) & 0xFF == 27:  # Tekan Esc untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
