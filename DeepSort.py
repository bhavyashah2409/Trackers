################# UNCOMMENT IF WANT TO USE DUPLICATE DEEPSORT LIBRARY

# import cv2 as cv
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# MAX_AGE = 60

# video = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_24.mp4"
# cap = cv.VideoCapture(video)

# model = YOLO('best.pt')
# tracker = DeepSort(max_age=MAX_AGE)

# while True:
#     ret, frame = cap.read()
#     detections = model.predict(frame)[0]
#     results = []
#     for xmin, ymin, xmax, ymax, p, c in detections.boxes.data.tolist():
#         results.append([[int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)], p, c])
#     tracks = tracker.update_tracks(results, frame=frame)
#     for track in tracks:
#         if track.is_confirmed():
#             i = track.track_id
#             c = track.get_det_class()
#             xmin, ymin, xmax, ymax = track.to_ltrb(orig=True)
#             frame = cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
#             p = track.get_det_conf()
#             if p is not None:
#                 frame = cv.putText(frame, f'ID: {i}, {c}: {round(p * 100, 2)}', (int(xmin), int(ymin - 10)), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
#             else:
#                 frame = cv.putText(frame, f'ID: {i}, {c}', (int(xmin), int(ymin - 10)), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
#     cv.imshow('Frame', frame)
#     cv.waitKey(1)
# cap.release()

################################ ORIGINAL YOLO + DEEPSORT

import cv2 as cv
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort import DeepSort

VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_40.mp4"
WEIGHTS = 'yolov8l.pt'

model = YOLO(WEIGHTS)
deepsort = DeepSort(
    # model_path=r'deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt (2).t7', # OLD, TRAINED ON MARKET1501
    model_path=r'deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt.t7', # NEW, TRAINED ON VIDEO DATASET VIDEOS
    max_dist=0.2,
    min_confidence=0.3,
    nms_max_overlap=0.5,
    max_iou_distance=0.7,
    max_age=70,
    n_init=5,
    nn_budget=100,
    use_cuda=True
    )
cap = cv.VideoCapture(VIDEO)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    preds = model.predict(frame)[0].cpu()
    classes = preds.names
    preds = preds.boxes
    xywhs = preds.xywh
    confs = preds.conf
    oids = preds.cls
    final = deepsort.update(xywhs, confs, oids, frame)
    if len(final) > 0:
        for xmin, ymin, xmax, ymax, i, c in final:
            frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            frame = cv.putText(frame, f'ID: {i}, {classes[c]}', (xmin, ymin + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    cv.imshow('Frame', frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
