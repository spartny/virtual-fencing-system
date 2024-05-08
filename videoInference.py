import cv2
from ultralytics import YOLO
from fenceBuilder import checkInside, fenceBuild, drawFence

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Open the video file

# home day time video
# video_path = "Videos\VIDEO_20240327_173849299.mp4"

# home night time video
# video_path = "Videos\VIDEO_20240327_173517186.mp4"

# multi angle video
# video_path = "Videos\VIDEO1.mp4"

# VIRAT dataset video
video_path = "Videos\VIRAT_S_010204_05_000856_000890.mp4"


cap = cv2.VideoCapture(video_path)

frame_count = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Define the Virtual Fence for the Scene 1st Frame only
        if frame_count == 0:
            fenceBuild(frame)
        frame_count += 1

        # every 20th frame is put through the YOLOv8 model
        if frame_count % 20 == 0:

            # draw fence over the frame
            drawFence(frame)
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.5, classes=[0], iou=0.7)
            boxes = results[0].boxes.xyxy.tolist()  # get the bounding box co-ords
            outcome = False
            for box in boxes:

                # calculating the center of the bounding boxes
                x_min, y_min, x_max, y_max = box
                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)
                
                # visualize the center of bounding box
                cv2.circle(frame, (center_x, center_y), 1, (0, 0, 255), 5)
                if outcome == False:

                    # checking whether centroid is inside of fence
                    outcome = checkInside(center_x, center_y)

            if outcome:
                cv2.putText(frame, 'BREACH', (300, 300), cv2.FONT_HERSHEY_SIMPLEX , 5, (0,0,255), 5, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'NO BREACH', (300, 300), cv2.FONT_HERSHEY_SIMPLEX , 5, (0,255,0), 5, cv2.LINE_AA)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()