import cv2
from ultralytics import YOLO
from fenceBuilder import checkInside, fenceBuild, drawFence


# Open the video file

# home day time video
video_path = "Videos\VIDEO_20240327_173849299.mp4"

# home night time video
# video_path = "Videos\VIDEO_20240327_173517186.mp4"

# multi angle video
# video_path = "Videos\VIDEO4.mp4"

# VIRAT dataset video
# video_path = "Videos\VIRAT_S_010204_05_000856_000890.mp4"

# Nighttime Video
# video_path = "Videos\\NighttimeVideo.mp4"

# load the YOLOv8 model
model = YOLO('yolov8m.pt')
cap = cv2.VideoCapture(video_path)

frame_count = 0
# loop through the video frames
while cap.isOpened():
    # read a frame from the video
    success, frame = cap.read()
    text_dim = int(len(frame) * 0.25)
    if success:
        # define the Virtual Fence for the Scene 1st Frame only
        if frame_count == 0:
            fenceBuild(frame)
        frame_count += 1

        # every 20th frame is put through the YOLOv8 model
        if frame_count % 10 == 0:

            # draw fence over the frame
            drawFence(frame)
            # run YOLOv8 inference on the frame
            results = model.track(frame, conf=0.6, classes=[0], persist=True, show=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes.xyxy.tolist()  # get the bounding box co-ords
            outcome = False
            for box in boxes:

                # calculating the feet of the bounding boxes
                x_min, y_min, x_max, y_max = box
                center_x = int((x_min + x_max) / 2)
                center_y = int(y_max)
                # center_y = int((2 * y_min * y_max )/ (y_min + y_max) 

                # visualize the center of bounding box
                cv2.circle(frame, (center_x, center_y), 7, (0,255,0), 8)
                if outcome == False:

                    # checking whether centroid is inside of fence
                    outcome = checkInside(center_x, center_y)

            if outcome:
                color = (0, 0, 255)
                cv2.putText(frame, 'BREACH', (text_dim, text_dim), cv2.FONT_HERSHEY_SIMPLEX , 5, (0,0,255), 5, cv2.LINE_AA)
            else:
                color = (0, 255, 0)
                cv2.putText(frame, 'NO BREACH', (text_dim, text_dim), cv2.FONT_HERSHEY_SIMPLEX , 5, (0,255,0), 5, cv2.LINE_AA)
            # visualize the results on the frame
            annotated_frame = results[0].plot()

            # display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # break the loop if the end of the video is reached
        break

# release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()