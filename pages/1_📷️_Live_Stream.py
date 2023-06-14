from settings import get_sit_up_with_weights, get_dumbbell_fly, get_barbell_curl, get_dumbbell_lateral_raise, get_seated_tricep_press, get_bent_over_dumbbell_row, get_squat_with_weights
from activity import Activity
from utils import get_mediapipe_pose
import av
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer

st.title('AI Weight Training Coach')

activity = st.radio(
    'Select Activity', ['Sit-up with Weights', 'Dumbbell Fly', 'Barbell Curl', 'Dumbbell Lateral Raise', 'Seated Tricep Press', 'Bent Over Dumbbell Row', 'Squat with Weights'], horizontal=True)

if activity == 'Sit-up with Weights':
    settings = get_sit_up_with_weights()
elif activity == 'Dumbbell Fly':
    settings = get_dumbbell_fly()
elif activity == 'Barbell Curl':
    settings = get_barbell_curl()
elif activity == 'Dumbbell Lateral Raise':
    settings = get_dumbbell_lateral_raise()
elif activity == 'Seated Tricep Press':
    settings = get_seated_tricep_press()
elif activity == 'Bent Over Dumbbell Row':
    settings = get_bent_over_dumbbell_row()
elif activity == 'Squat with Weights':
    settings = get_squat_with_weights()


live = Activity(settings=settings, flip_frame=True)
pose = get_mediapipe_pose()


def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format='rgb24')  # Decode and get RGB frame
    if activity == 'Sit-up with Weights':
        frame, _ = live.process_sit_up_with_weights(frame, pose)
    elif activity == 'Dumbbell Fly':
        frame, _ = live.process_dumbbell_fly(frame, pose)
    elif activity == 'Barbell Curl':
        frame, _ = live.process_barbell_curl(frame, pose)
    elif activity == 'Dumbbell Lateral Raise':
        frame, _ = live.process_dumbbell_lateral_raise(frame, pose)
    elif activity == 'Seated Tricep Press':
        frame, _ = live.process_seated_tricep_press(frame, pose)
    elif activity == 'Bent Over Dumbbell Row':
        frame, _ = live.process_bent_over_dumbbell_row(frame, pose)
    elif activity == 'Squat with Weights':
        frame, _ = live.process_squat_with_weights(frame, pose)
    # Encode and return BGR frame
    return av.VideoFrame.from_ndarray(frame, format='rgb24')


ctx = webrtc_streamer(
    key='ai-weight-training-coach',
    video_frame_callback=video_frame_callback,
    rtc_configuration={'iceServers': [
        {'urls': ['stun:stun.1.google.com:19302']}
    ]},
    media_stream_constraints={
        'video': {'width': {'min': 640, 'ideal': 640}}, 'audio': False
    },
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True, controls=False, width=720, muted=False
    )
)
