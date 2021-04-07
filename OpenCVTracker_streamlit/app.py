import cv2
import streamlit as st
import tempfile
import imutils
from PIL import Image
import pandas as pd
from imutils.video import FPS
import time
from random import randint
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from streamlit_drawable_canvas import st_canvas
import OpenCVTracker_streamlit.SessionState as SessionState
import streamlit.report_thread as ReportThread
from streamlit.server.server import Server

def trigger_rerun():
    """
    mechanism in place to force resets and update widget states
    """
    session_infos = Server.get_current()._session_info_by_id.values()
    for session_info in session_infos:
        this_session = session_info.session
    this_session.request_rerun()

def app_tracker():
    OPENCV_Trackers = {
        "boost" : cv2.TrackerBoosting_create(),
        "mil" : cv2.TrackerMIL_create(),
        "kcf" : cv2.TrackerKCF_create(),
        "tld" : cv2.TrackerTLD_create(),
        "median" : cv2.TrackerMedianFlow_create(),
        "goturn" : cv2.TrackerGOTURN_create(),
        "crst" : cv2.TrackerCSRT_create(),
        "moose" : cv2.TrackerMOSSE_create()
    }

    DESCIPTIONS = {"boost": "__Boosting Tracker__: This algorithm is an online version of AdaBoost "
                            "(the algorithm that HAAR cascade based). The initial bounding box will be provided "
                            "from the user or from another object detection model, it will be used as a positive "
                            "sample of the classifier, then the other image patches will be seen as negative samples. "
                            "Given a new image, the algorithm will look at each neighbor pixel of the location provided "
                            "at the previous frame, and score this by the classifier. the new location of the object "
                            "will be the one having the highest score. As more frames come in, the more data we have. "
                            "\n - __Pros__: None\n - __Cons__: Mediocre",
                   "mil": "__MIL Tracker (Multiple Instance Learning)__: The idea is the same with the Boosting Tracker; "
                          "however, the main difference is that in MIL tracker, we do not have only one positive sample "
                          "in a frame, the algorithm also looks at some neighbor patches as positive ones. "
                          "This is for the case that even if the current location of the image is not correct, "
                          "it can contain one of the correct one in those positive samples. "
                          "\n - __Pros__: Work quite well for partial occlusion."
                          "\n - __Cons__: Cannot recover from full occlusion. Fail tracker cannot be reported properly.",
                   "kcf": "__KCF Tracker (Kernelized Correlation Filters)__: The tracker will utilize the overlapping positive "
                          "regions from MIL to make tracking faster and more accurate at the same time."
                          "\n - __Pros__: Accuracy and speed are both better than MIL and Boosting and better in reporting tracking failure."
                        "\n - __Cons__: Does not recover from full occlusion",
                   "tld": "__TLD Tracker (Tracking Learning and detection)__: TLD stands for tracking, learning, and detecting. "
                          "The detector will localizes all appearance having been tracking so far and correct the tracker if necessary. "
                          "Then the learning will estimate the detector error and update the detector."
                            "\n - This algorithm can recover if the object is being occluded and also invariant "
                          "with the scale changes, because of detector. However, the algorithms make lots of false positive "
                          "even false negative, makes it also unusable",
                   "median": "__MedianFlow__: Consider both forward and backward direction of object’s trajectory, "
                             "and based on the ForwardBackward error to update the learner."
                             "\n - __Pros__: good at reporting fail tracking."
                             "\n - __Cons__: Only work well for slow movement object (small motion between two frames)",
                   "goturn": "__GOTURN__: based on CNN to update bounding box"
                             "\n - __Pros__: Robust to the viewpoint change, lighting change, and deformations."
                             "\n - __Cons__: Cannot deal with occlusion well.",
                   "crst": "__CSRT Tracker__: The tracker is applied the Discriminative Correlation Filter with Channel and "
                           "Spatial Reliability published in this paper: https://arxiv.org/abs/1611.08461 to enlarging "
                           "and localization of the selected region improved tracking of non-rectangular regions or objects"
                           "\n-  It also operates at a comparatively lower fps (25fps) but gives higher accuracy for object tracking. "
                           "\n- cannot recover from full occlusion "
                           "\n- Working very well with scale variance and partial occlusion"
                           "\n- Cannot report the failure ",
                   "moose": "__Moose tracker (Minimum Output Sum of Squared Error)__: Robust to variations in lighting, scale, "
                            "pose, and non-rigid deformations.It also detects occlusion, which enables the tracker to pause and resume where it left off"
                            "\n- Fast (450fps) As accurate as other methods but not better than KCF and CSRT "
                            "\n - can report when it fail"}

    st.header("Opencv Object Trackers")
    st.write("My own experiment notes: :sunglasses:"
             "\n - __KCF tracker__: Fast (> 100 fps) well reported, high accuracy, Cannot recover from occulations"
             "\n - __Mosse__: Very Faster but still main high accuracy still less than KCF or CSRT "
             "\n - __CSRT__: slower than KCF and Mosse just 20-40 fps in CPU but open very high accuracy"
             "\n - While the other methods the basic need is that the object must be separated from the background")
    state = SessionState.get(upload_key=None, enabled=True, start_track=False, start_choose=False, chosen=False, run=False,
                             canvas_result=None)
    choose_obj = st.empty()
    chosen = st.empty()
    start_track_button = st.empty()
    start_choose_button = st.empty()
    stop_button = st.empty()
    stframe = st.empty()
    initBB = None
    fps = None
    vs = None
    # Choosing type of tracker
    type = st.sidebar.selectbox("Algorithms:", ("none", "boost", "mil", "kcf", "tld", "median", "goturn", "crst", "moose"))
    # Define tracker
    if type != "none":
        tracker = OPENCV_Trackers[type]
        st.write(DESCIPTIONS[type])
    # upload video to process

    f = st.file_uploader("Video", type=["mp4"])
    if f is not None:

        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(f.read())
        vs = cv2.VideoCapture(tfile.name)

        vid = cv2.VideoCapture(tfile.name)
        if not state.run:
            start_choose = start_choose_button.checkbox("Start Choose Object")
            state.start_choose = start_choose
        if state.start_choose:
            grabbed, frame = vs.read()
            frame = imutils.resize(frame, width=500)
            H, W = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            canvas_result = st_canvas(fill_color="rgba(255,165,0,0.3)", stroke_width=3, background_image=pil_image,
                                          height=H, width=W, drawing_mode="rect", key="canvas", update_streamlit=False)
            state.canvas_result = canvas_result
            if state.canvas_result.json_data is not None:
                st.dataframe(pd.json_normalize(state.canvas_result.json_data["objects"]))
                data = pd.json_normalize(state.canvas_result.json_data["objects"])
                data = data.iloc[-1]
                x = int(data["left"])
                y = int(data['top'])
                w = int(data['width'])
                h = int(data['height'])
                initBB = (x, y, w, h)
                fps = FPS().start()
                tracker.init(frame, initBB)
            vs.release()
            start_track = start_track_button.checkbox("Start Tracking")
            state.start_track = start_track
            if state.start_track:
                start_choose_button.empty()
                start_track_button.empty()
                state.enabled = False
                stop = stop_button.button("Stop")
                if state.run:
                    tfile.close()
                    f.close()
                    state.upload_key = str(randint(1000, int(1e6)))
                    state.enabled = True
                    state.run = False
                    while True:
                        # stframe.empty()
                        grabbed, frame = vid.read()

                        if not grabbed:
                            st.write("Cannot receive frame... Exiting!!!")
                            st.stop()

                        if stop:
                            st.stop()
                            break
                        frameid = int(round(vid.get(1)))

                        if frameid == 1:
                            continue
                        frame = imutils.resize(frame, width=500)
                        H, W = frame.shape[:2]
                        if initBB is not None:
                            success, box = tracker.update(frame)
                            if success:
                                (x,y,w,h) = [int(v) for v in box]
                                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
                            fps.update()
                            fps.stop()
                            info = [
                                ("Tracker", type),
                                ("Success", success),
                                ("FPS", "{:.2f}".format(fps.fps()))
                            ]
                            for i, (k,v) in enumerate(info):
                                text = "{}: {}".format(k, v)
                                cv2.putText(frame, text, (20,i*20+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
                        frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        stframe.image(frm)
                else:
                    state.run = True
                    trigger_rerun()


        vid.release()



