import cv2


video_capture = cv2.VideoCapture("Videos\VIDEO_20240327_173849299.mp4") 

if not video_capture.isOpened():
    print("Unable to open video stream")
    exit()

frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("End of Video")
        break

    frame_count += 1

    if frame_count % 10 == 0:
        cv2.imwrite(f"frame_{frame_count}.jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
