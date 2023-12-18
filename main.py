import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import torch
import time

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    print("Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


model = YOLO('best_4.pt')
label = model.names
# print(label)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cctv1.mp4')
# Ubah titik-titik sudut menjadi array NumPy
roi_points = np.array(
    [[575, 300], [700, 300], [790, 470], [580, 470]], np.int32)
roi_points1 = np.array(
    [[570, 210], [650, 210], [700, 300], [575, 300]], np.int32)
roi_points2 = np.array(
    [[565, 120], [605, 120], [650, 210], [570, 210]], np.int32)

# vidoe2
# roi_points = np.array(
#     [[490, 230], [610, 230], [700, 387], [510, 389]], np.int32)
# roi_points1 = np.array(
#     [[480, 130], [550, 130], [610, 230], [490, 230]], np.int32)
# roi_points2 = np.array(
#     [[474, 80], [523, 80], [550,130 ], [480, 130]], np.int32)

# Get the frames per second (fps) and frame dimensions of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280,720))  # Adjust parameters as needed
# print(cv2.getBuildInformation())
if not out.isOpened():
    print("Error: VideoWriter not opened.")

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))

    results = model.predict(frame, device=device, verbose=False)

    a = results[0].cpu().numpy().boxes.data

    px = pd.DataFrame(a).astype("float")

    list9 = []
    car_roi = {'car_roi1': 0, 'car_roi2': 0, 'car_roi3': 0}
    motor_roi = {'motor_roi1': 0, 'motor_roi2': 0, 'motor_roi3': 0}
    truck_roi = {'truck_roi1': 0, 'truck_roi2': 0, 'truck_roi3': 0}
    bus_roi = {'bus_roi1': 0, 'bus_roi2': 0, 'bus_roi3': 0}

    for index, row in px.iterrows():

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        point = (cx, cy)

        # roi 1
        results9 = cv2.pointPolygonTest(roi_points, point, False)
        if results9 >= 0:
            if d == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                bus_roi['bus_roi1'] = bus_roi['bus_roi1'] + 1
            elif d == 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                car_roi['car_roi1'] = car_roi['car_roi1'] + 1
            elif d == 2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                motor_roi['motor_roi1'] = motor_roi['motor_roi1'] + 1
            elif d == 4:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                truck_roi['truck_roi1'] = truck_roi['truck_roi1'] + 1
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # roi 2
        results9 = cv2.pointPolygonTest(roi_points1, point, False)
        if car_roi['car_roi1'] >= 2 or bus_roi['bus_roi1'] >= 1 or truck_roi['truck_roi1'] >= 1 or motor_roi['motor_roi1'] >= 5 or (car_roi['car_roi1'] >= 1 and motor_roi['motor_roi1'] >= 3):
            if results9 >= 0:
                if d == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    bus_roi['bus_roi2'] = bus_roi['bus_roi2'] + 1
                elif d == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    car_roi['car_roi2'] = car_roi['car_roi2'] + 1
                elif d == 2:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    motor_roi['motor_roi2'] = motor_roi['motor_roi2'] + 1
                elif d == 4:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    truck_roi['truck_roi2'] = truck_roi['truck_roi2'] + 1
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # roi 3
        results9 = cv2.pointPolygonTest(roi_points2, point, False)
        if car_roi['car_roi2'] >= 2 or bus_roi['bus_roi2'] >= 1 or truck_roi['truck_roi2'] >= 1 or motor_roi['motor_roi2'] >= 5 or (car_roi['car_roi2'] >= 1 and motor_roi['motor_roi2'] >= 3):
            if results9 >= 0:
                if d == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    bus_roi['bus_roi3'] = bus_roi['bus_roi3'] + 1
                elif d == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    car_roi['car_roi3'] = car_roi['car_roi3'] + 1
                elif d == 2:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    motor_roi['motor_roi3'] = motor_roi['motor_roi3'] + 1
                elif d == 4:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    truck_roi['truck_roi3'] = truck_roi['truck_roi3'] + 1
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    if (car_roi['car_roi3'] >= 3 or bus_roi['bus_roi3'] >= 1 or truck_roi['truck_roi3'] >= 1 or motor_roi['motor_roi3'] >= 5 or (car_roi['car_roi3'] >= 2 and motor_roi['motor_roi3'] >= 3)):
        cv2.putText(frame, f"Durasi Lampu Merah  = 50s", (
            650, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    elif (car_roi['car_roi2'] >= 3 or bus_roi['bus_roi2'] >= 1 or truck_roi['truck_roi2'] >= 1 or motor_roi['motor_roi2'] >= 5 or (car_roi['car_roi2'] >= 2 and motor_roi['motor_roi2'] >= 3)):
        cv2.putText(frame, f"Durasi Lampu Merah  = 60s", (
            650, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    else:
        cv2.putText(frame, f"Durasi Lampu Merah  = 70s", (
            650, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)

    cv2.polylines(frame, [roi_points], isClosed=True,
                  color=(0, 0, 255), thickness=2)
    cv2.polylines(frame, [roi_points1], isClosed=True,
                  color=(255, 0, 255), thickness=2)
    cv2.polylines(frame, [roi_points2], isClosed=True,
                  color=(0, 255, 255), thickness=2)

    cv2.putText(frame, f"motor = {motor_roi['motor_roi1']}", (
        1100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"car    = {car_roi['car_roi1']}",
                (1100, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"bus    = {bus_roi['bus_roi1']}",
                (1100, 70), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"truck  = {truck_roi['truck_roi1']}", (
        1100, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)

    cv2.putText(frame, f"motor = {motor_roi['motor_roi2']}", (
        1100, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"car    = {car_roi['car_roi2']}", (
        1100, 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"bus    = {bus_roi['bus_roi2']}", (
        1100, 170), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"truck  = {truck_roi['truck_roi2']}", (
        1100, 190), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)

    cv2.putText(frame, f"motor = {motor_roi['motor_roi3']}", (
        1100, 230), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"car    = {car_roi['car_roi3']}", (
        1100, 250), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"bus    = {bus_roi['bus_roi3']}", (
        1100, 270), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
    cv2.putText(frame, f"truck  = {truck_roi['truck_roi3']}", (
        1100, 290), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)

    # Calculate fps
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    start_time = time.time()

    # Display fps on the top left corner
    cv2.putText(frame, f"FPS: {int(fps)}", (100, 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    
    # Write the frame to the output video file
    out.write(frame)
    
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
