import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')
video_path = r'C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_24.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, conf=0.65, iou=0.8, persist=True)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
