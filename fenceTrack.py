import cv2
from ultralytics import YOLO
from fenceBuilder import checkInside, fenceBuild, drawFence, update_coordinates, calculate_angle_vector
import torch


# Check if a GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Open the video file

# home day time video
# video_path = "Videos\\VIDEO_20240327_173849299.mp4"

# home night time video
# video_path = "Videos\\VIDEO_20240327_173517186.mp4"

# multi angle video
video_path = "Videos\\VIDEO4.mp4"

# VIRAT dataset video
# video_path = "Videos\\VIRAT_S_010204_05_000856_000890.mp4"

# Nighttime Video
# video_path = "Videos\\NighttimeVideo.mp4"

# load the YOLOv11m model
model = YOLO('yolov8m.pt').to(device)
cap = cv2.VideoCapture(video_path)

frame_count = 0

past_coordinates = dict()

# loop through the video frames
while cap.isOpened():
    # read a frame from the video
    success, frame = cap.read()
    text_dim = int(len(frame) * 0.25) if frame is not None else 200
    
    if success:
        # define the Virtual Fence for the Scene 1st Frame only
        if frame_count == 0:
            fenceBuild(frame)
        frame_count += 1

        # every 20th frame is put through the YOLOv8 model
        # if frame_count % 10 == 0:
        if frame_count:
            # draw fence over the frame
            drawFence(frame)

            # get fence values from saved .txt file
            file = open('fence.txt', 'r')
            fence = eval(file.read())
            file.close()
            
            # run YOLO inference on the frame
            results = model.track(frame, conf=0.4, classes=[0], persist=True, tracker="bytetrack.yaml")
            
            for result in results:
                boxes = result.boxes
                outcome = False

                # visualize the results on the frame
                annotated_frame = result.plot()

                for box in boxes:
                    box_coords = box.xyxy[0]  # get the bounding box coordinates
                    track_id = box.id                # gget the track ID

                    # calculating the feet of the bounding boxes
                    x_min, y_min, x_max, y_max = box_coords
                    foot_x = int((x_min + x_max) / 2)
                    foot_y = int(y_max)
                    
                    # gget the current coordinates
                    current_coords = (foot_x, foot_y)

                    # update past coordinates for the track ID
                    update_coordinates(track_id, current_coords, past_coordinates)
                    angle = calculate_angle_vector(track_id, past_coordinates)
                    print(track_id, '-->', angle)

                    # visualize the center of bounding box
                    cv2.circle(annotated_frame, (foot_x, foot_y), 2, (0,255,0), 5)
                    if outcome == False:

                        # checking whether centroid is inside of fence
                        outcome = checkInside(foot_x, foot_y, fence, track_id, past_coordinates, danger_vector=270)

            if outcome:
                color = (0, 0, 255)
                cv2.putText(annotated_frame, 'BREACH', (text_dim, text_dim), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,0,255), 2, cv2.LINE_AA)
            else:
                color = (0, 255, 0)
                cv2.putText(annotated_frame, 'NO BREACH', (text_dim, text_dim), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 2, cv2.LINE_AA)
            

            # Resize the annotated frame back to the original frame dimensions
            annotated_frame_resized = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

            # display the annotated frame
            # cv2.imshow("YOLO Inference", cv2.cvtColor(annotated_frame_resized, cv2.COLOR_BGR2RGB))
            cv2.imshow("YOLO Inference", annotated_frame_resized)

            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # break the loop if the end of the video is reached
        break

# release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()