from PeopleCounter.TrackingObject.centroidtracker import CentroidTracker
from PeopleCounter.TrackingObject.TrackableObject import TrackableObject

import cv2
import imutils
import numpy as np
import streamlit as st
import tempfile
import dlib
def hide_streamlit_widgets():
    """
    hides widgets that are displayed by streamlit when running
    """
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
def app_count_people():

    st.title("People Counter: ")
    st.write("In this tutorial, I apply object tracking technique to implement a people counter model. The model using a simple "
             "technique is centroid tracker that re-assigns the new centroid of the object ID based on the distance in each frame"
             "That is followed by correlation tracker algorithm to update the centroid according to the next frame. "
             )
    st.subheader("1. Centroid Tracking: ")
    st.write("- __Step 1__: Accept bounding box coordinates and compute centroids:"
             "\n  + For every frame of video we need to provide the bounding boxes for the objects in current frame "
             "(the object detector can be any model you like, HOGs, FasterRCNN,...), then basing on these bounding boxes, we compute centroids for each."
             "\n  + At the initial frame, besides feeding the bounding boxes to the tracker, we need to assign them a particular identifier for each to perform tracking later."
             "\n- __Step2__: Re-identification:"
             "\n  + In the current frame, we still need to provide bounding boxes, and the according new centroids to "
             "the tracker. However, one thing different is that we need to re-assign the id for each object again "
             "(which is actually the main purpose of any tracking algorithms). "
             "\n  + The ideal is we compute the euclidean distances between the centroid of current centroids and "
             "those in the previous frame. The Id will be assigned to the closest one."
             "\n  + Finally, we update (x,y) coordinates of existing objects; however, we may face a problem that "
             "the new object will appear like in the fig2. May be it can be a false positive but that depends on your detectors. "
             "In this case, the tracker will assign new ID for this new centroids (represents to the new objects appears)"
             "\n- __Step3__: How about when an object disappears, we need to deregister old objects."
             "\n  + we will deregister the old object when they can not match any existing objects for total N subsequence frames")
    st.subheader("2. Correlation Tracker: ")
    st.write("- The algorithm is from the paper (http://www.bmva.org/bmvc/2014/files/paper038.pdf) of Danelljian et al.’s "
             "with approach for robust scale estimation in a tracking-by-detection framework. "
             "The proposed approach works by learning discriminative correlation filters based on scale pyramids"
             "\n- __Performance__: "
             "\n  + The tracker achieves a quite high speed of 40 fps "
             "\n  + We cannot report if the tracker fails in tracking right object "
             "\n  + The tracker will totally depend on the performance of your object detection. "
             "\n  + Working well when overlap"
             "\n - However, in the case the tracker loses the object. We can use the detector again to reidentify objects.")

    MODEL_URL = "/home/tuyen/ObjectTracking-Tutorials/PeopleCounter/mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
    PROTXT_URL = "/home/tuyen/ObjectTracking-Tutorials/PeopleCounter/mobilenet_ssd/MobileNetSSD_deploy.prototxt"

    net = cv2.dnn.readNetFromCaffe(PROTXT_URL, MODEL_URL)

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    hide_streamlit_widgets()
    conf = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, step=0.1)
    skipframe = st.sidebar.slider("The number of frame to rerun detector", min_value=0, max_value=30, step=5)
    maxDistance = st.sidebar.slider("Max Distance to assign object in subsequence frame", min_value=30, max_value=70, step=10)
    maxDisappear = st.sidebar.slider("Max Disappear frames of objects to be removed", min_value=30, max_value=70, step=10)
    totalFrame = 0
    totalUp = 0
    totalDown = 0
    trackers = []
    status = ""
    ct = CentroidTracker(maxDistance, maxDisappear)
    trackable_object_dict = {}
    start_button = st.empty()
    stop_button = st.empty()
    total_Ups = st.empty()
    total_Downs = st.empty()
    status_space = st.empty()
    stframe = st.empty()
    f = st.file_uploader("Upload your video", type=['mp4', "avi"])
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(f.read())
        vs = cv2.VideoCapture(tfile.name)

        _start = start_button.button("Start")

        if _start:
            _stop = stop_button.button("Stop")
            while True:
                grabbed, frame = vs.read()
                if frame is None:
                    st.write("No more frame is detected. Exitting!!!")
                    break
                if _stop:
                    break
                frame = imutils.resize(frame, width=500)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                H, W = frame.shape[:2]
                rects = []
                if totalFrame % skipframe == 0:
                    status = "Detecting"
                    trackers = []
                    blob = cv2.dnn.blobFromImage(frame, 1/127.5, (W, H), 127.5)
                    net.setInput(blob)
                    detections = net.forward()

                    for i in range(detections.shape[2]):
                        confidence = detections[0,0,i,2]
                        if confidence >= conf:
                            class_idx = int(detections[0,0,i,1])

                            if CLASSES[class_idx] == "person":
                                box = detections[0,0,i,3:7] * np.array([W, H, W, H])
                                (x0, y0, x1, y1) = box.astype("int")

                                tracker = dlib.correlation_tracker()
                                rect = dlib.rectangle(x0,y0, x1, y1)
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

                for objectID, centroid in objects.items():
                    to = trackable_object_dict.get(objectID, None)
                    if to is None:
                        to = TrackableObject(objectID, centroid)
                    else:
                        if not to.counted:
                            if len(to.centroids) >= 5:
                                y = [c[1] for c in to.centroids]

                                direction = centroid[1] - np.mean(y)

                                if direction > 0 and centroid[1] > H//2:
                                    totalDown += 1
                                    to.counted = True
                                if direction < 0 and centroid[1] < H//2:
                                    totalUp += 1
                                    to.counted = True
                        to.centroids.append(centroid)
                    trackable_object_dict[objectID] = to
                    cv2.circle(frame, (centroid[0], centroid[1]), radius=3, color=(0,255,0), thickness=-1)
                    cv2.putText(frame, "ID {}".format(objectID), (centroid[0], centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,255,0),2, True)
                frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frm)
                total_Ups.markdown("Total Ups: {}".format(totalUp))
                total_Downs.markdown("Total Downs: {}".format(totalDown))
                status_space.markdown("Status: {}".format(status))
                totalFrame += 1
            vs.release()






