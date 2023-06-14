from settings import get_sit_up_with_weights, get_dumbbell_fly, get_barbell_curl, get_dumbbell_lateral_raise, get_seated_tricep_press, get_bent_over_dumbbell_row, get_squat_with_weights
from activity import Activity
from utils import get_mediapipe_pose
import os
import sys
import streamlit as st
import cv2
import tempfile

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

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


upload = Activity(settings=settings, flip_frame=True)
pose = get_mediapipe_pose()

download = None

if 'download' not in st.session_state:
    st.session_state['download'] = False


output_video_file = f'output_recorded.mp4'

if os.path.exists(output_video_file):
    os.remove(output_video_file)


with st.form('Upload', clear_on_submit=True):
    up_file = st.file_uploader("Upload a Video", ['mp4', 'mov', 'avi'])
    uploaded = st.form_submit_button("Upload")

stframe = st.empty()

ip_vid_str = '<p style="font-family:Helvetica; font-weight: bold; font-size: 16px;">Input Video</p>'
warning_str = '<p style="font-family:Helvetica; font-weight: bold; color: Red; font-size: 17px;">Please Upload a Video first!!!</p>'

warn = st.empty()


download_button = st.empty()

if up_file and uploaded:

    download_button.empty()
    tfile = tempfile.NamedTemporaryFile(delete=False)

    try:
        warn.empty()
        tfile.write(up_file.read())

        vf = cv2.VideoCapture(tfile.name)

        # ---------------------  Write the processed video frame. --------------------
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(
            output_video_file, fourcc, fps, frame_size)
        # -----------------------------------------------------------------------------

        txt = st.sidebar.markdown(ip_vid_str, unsafe_allow_html=True)
        ip_video = st.sidebar.video(tfile.name)

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            # convert frame from BGR to RGB before processing it.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if activity == 'Sit-up with Weights':
                out_frame, _ = upload.process_sit_up_with_weights(frame, pose)
            elif activity == 'Dumbbell Fly':
                out_frame, _ = upload.process_dumbbell_fly(frame, pose)
            elif activity == 'Barbell Curl':
                out_frame, _ = upload.process_barbell_curl(frame, pose)
            elif activity == 'Dumbbell Lateral Raise':
                out_frame, _ = upload.process_dumbbell_lateral_raise(
                    frame, pose)
            elif activity == 'Seated Tricep Press':
                out_frame, _ = upload.process_seated_tricep_press(frame, pose)
            elif activity == 'Bent Over Dumbbell Row':
                out_frame, _ = upload.process_bent_over_dumbbell_row(
                    frame, pose)
            elif activity == 'Squat with Weights':
                out_frame, _ = upload.process_squat_with_weights(frame, pose)
            stframe.image(out_frame)
            video_output.write(out_frame[..., ::-1])

        vf.release()
        video_output.release()
        stframe.empty()
        ip_video.empty()
        txt.empty()
        tfile.close()

    except AttributeError:
        warn.markdown(warning_str, unsafe_allow_html=True)


if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button(
            'Download Video', data=op_vid, file_name='output_recorded.mp4')

    if download:
        st.session_state['download'] = True


if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()
