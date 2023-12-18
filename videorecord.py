import cv2

# Open a video capture object (0 for default camera)
cap = cv2.VideoCapture(0)

# Get the default video dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object
# out = cv2.VideoWriter('output1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))  # Adjust 

# Define the codec and create a VideoWriter object using FFMPEG backend
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'h264'
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))  # Adjust parameters as needed
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))  # Adjust 

# Check if VideoWriter is opened successfully
if not out.isOpened():
    print("Error: VideoWriter not opened.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Write the frame to the output video file
    out.write(frame)

    cv2.imshow('Frame', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
