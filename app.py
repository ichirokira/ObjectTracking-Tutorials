from OpenCVTracker_streamlit.app import app_tracker
from PeopleCounter.app_count_people import app_count_people

import streamlit as st

st.title(" Object Tracking Tutorials ")
st.header("About")
st.write("Object tracking is the task of taking an initial set of object detections, creating a unique ID for each of "
         "the initial detections, and then tracking each of the objects as they move around frames in a video, "
         "maintaining the ID assignment")
st.write("In this tutorials, I give you a demo for different object tracking algorithms also their own characteristic "
         "to apply your own application. Those implemented algorithms either from OpenCV libraries or some state-of-the-art method "
         "using Deep Learning. Finally, I show one application of object tracking task that is People counters.")
page = st.selectbox("Search", ("None","Opencv Trackers", "Deep Learning-based Tracker", "People Counter Application"))

if page == "Opencv Trackers":
    app_tracker()
elif page == "People Counter Application":
    app_count_people()