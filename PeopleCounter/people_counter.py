import argparse
import cv2
import dlib
import imutils
import numpy as np
from imutils.video import VideoStream, FPS
from TrackingObject.centroidtracker import CentroidTracker
from TrackingObject.TrackableObject import TrackableObject

ap = argparse.ArgumentParser()
ap.add_argument("--model", "-m", default=None, required=True, type=str, help="Path to detector model")
ap.add_argument("--protxt", "-p", default=None, required=True, type=str, help="Path to protxt file")
ap.add_argument("--output", "-o", default=None, required=True, type=str, help="Path to output file")
ap.add_argument("--video", "-v", default=None, required=True, type=str, help="Path to the input video else the system will access your camera")
ap.add_argument("--conf", "-c", default=0.4, type=float, help="Confidence of the detector")
ap.add_argument("--skipframes", "-s", default=30, type=int, help="Rerun detector after running a particular number of frames")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

writer = None
net = cv2.dnn.readNetFromCaffe(args["protxt"], args['model'])
fps = None
total_frame = 0
totalUps = 0
totalDowns = 0
trackers = []
status = ""
ct = CentroidTracker(maxDistance=50, maxDisappear=40)
trackable_object = {}
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

while True:
    frame = vs.read()
    frame = frame if not args.get("video", False) else frame[1]
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H,W = frame.shape[:2]
    rects = []
    if total_frame % args["skipframes"] == 0:
        status = "Detecting"
        trackers = []
        blob = cv2.dnn.blobFromImage(frame, 1/127.5, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0,0,i,2]

            #print("{}, {}: {}".format(total_frame, CLASSES[class_id], confidence))
            if confidence > args["conf"]:
                idx = int(detections[0,0,i,1])
                if CLASSES[idx] != 'person':
                    continue
                box = detections[0,0,i,3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)

    else:
        status = "Waiting"

        for tracker in trackers:
            status = "Tracking"
            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            rects.append([startX, startY, endX, endY])
    objects = ct.update(rects)

    for id, centroid in objects.items():
            to = trackable_object.get(id, None)
            if to is None:
                to = TrackableObject(id, centroid)
            else:
                if len(to.centroids) >5:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)

                    if not to.counted:
                            if direction < 0 and centroid[1] < H//2:
                                totalUps += 1
                                to.counted = True
                            elif direction > 0 and centroid[1] > H//2 :
                                totalDowns += 1
                                to.counted = True
                to.centroids.append(centroid)
            trackable_object[id] = to
            cv2.circle(frame, (centroid[0], centroid[1]), 2, (0,255,0), -1)
            cv2.putText(frame,"ID {}".format(id), (centroid[0]-10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0)
                        , 2, True)
    total_frame += 1
    cv2.line(frame, (0, H//2), (W, H//2), (0,168,168),2, True)

    text = [("TotalUps: ", totalUps), ("TotalDowns: ", totalDowns), ("Status: ", status)]

    for i, (k, v) in enumerate(text):
        cv2.putText(frame, k+"{}".format(v), (0,i*20+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Frame", frame)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
    if writer is not None:
        writer.write(frame)

    key = cv2.waitKey(1) or 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
if not args.get("video", False):
    vs.stop()
else:
    vs.release()
if writer is not None:
    writer.release()