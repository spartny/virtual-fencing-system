import cv2
from ultralytics import YOLO
from fenceBuilderTest import checkInside, fenceBuild, drawFence, update_coordinates, calculate_angle_vector
import torch
import math
import numpy as np

# Check if a GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Open the video file

# home day time video
# video_path = "Videos\\VIDEO_20240327_173849299.mp4"

# home night time video
# video_path = "Videos\\VIDEO_20240327_173517186.mp4"

# multi angle video
# video_path = "Videos\\VIDEO4.mp4"

# VIRAT dataset video
#video_path = "Videos\\VIRAT_S_010204_05_000856_000890.mp4"

# Nighttime Video
video_path = "Videos\\NighttimeVideo.mp4"
# video_path = "Videos\\WhatsApp Video 2024-05-15 at 22.04.33_2173c695.mp4"
# load the YOLOv11m model
model = YOLO('yolov8m.pt').to(device)
cap = cv2.VideoCapture(video_path)

frame_count = 0

past_coordinates = dict()
def calculate_angle_vector(track_id, past_coordinates, reference='x'):
    track_id_int = int(track_id.item()) if track_id is not None else 'None'
    if track_id_int not in past_coordinates or len(past_coordinates[track_id_int]) < 2:
        return -1.0

    coords = past_coordinates[track_id_int]  # Last 10 positions
    total_dx, total_dy = 0, 0

    # Calculate total change
    for i in range(1, len(coords)):
        previous_coords = coords[i - 1]
        current_coords = coords[i]
        total_dx += current_coords[0] - previous_coords[0]
        total_dy += current_coords[1] - previous_coords[1]

    avg_dx = total_dx / (len(coords) - 1)
    avg_dy = total_dy / (len(coords) - 1)

    if reference == 'x':
        angle = math.atan2(avg_dy, avg_dx)
    elif reference == 'y':
        angle = math.atan2(avg_dx, avg_dy)
    else:
        raise ValueError("Reference must be 'x' or 'y'")

    angle_degrees = math.degrees(angle) % 360

    # Normalize angle vector for consistency
    angle_degrees = (angle_degrees + 360) % 360
    return angle_degrees

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
                    center_x = int((x_min + x_max) / 2)
                    center_y = int((y_min + y_max) / 2)
                    current_points = [
                        (foot_x-5, foot_y),         # Top-left
                        (foot_x, foot_y-5),
                        (foot_x+5, foot_y),        # Top-right
                        (foot_x-5, foot_y-5), 
                        (foot_x+5, foot_y-5),        # Bottom-left
                        (center_x-5, center_y-5),
                        (center_x-5, center_y),
                        (center_x, center_y-5),
                        (center_x, center_y+5),
                        (center_x+5, center_y+5),
                        (center_x-5, center_y+5),
                        (center_x+5, center_y-5),
                        (center_x, center_y),  
                        (center_x+5, center_y),            # Center
                        (foot_x, foot_y)
                    ] 
                    # gget the current coordinates
                    #current_coords = (foot_x, foot_y)
                    angle_dict={}

                    for current_coords in current_points:
                        # update past coordinates for the track ID
                        update_coordinates(track_id, current_coords, past_coordinates)
                        angle = calculate_angle_vector(track_id, past_coordinates)
                        if track_id not in angle_dict:
                            angle_dict[track_id] = []
                        angle_dict[track_id].append(angle) 
                    average_angle = np.average(angle_dict[track_id])
                    print(track_id, '-->', average_angle)
                    # print(past_coordinates)
                    print(angle_dict)
                    # visualize the center of bounding box
                    cv2.circle(annotated_frame, (foot_x, foot_y), 2, (0,255,0), 5)
                    if outcome == False:

                        # checking whether centroid is inside of fence
                        outcome = checkInside(foot_x, foot_y,track_id, fence, average_angle, danger_vector=180)

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