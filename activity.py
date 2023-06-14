import time
import cv2
import numpy as np
from utils import find_angle, find_dist, get_landmark_features, draw_text, draw_dotted_line, get_visibility
from pygame import mixer

class Activity:
    def __init__(self, settings, flip_frame=False):

        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame

        self.settings = settings

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        self.prev_frame_time = 0
        self.new_frame_time = 0

        # Colors in RGB format.
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255)
        }

        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
            'ear': 7,
            'shoulder': 11,
            'elbow': 13,
            'wrist': 15,
            'hip': 23,
            'knee': 25,
            'ankle': 27,
            'foot': 31
        }

        self.right_features = {
            'ear': 8,
            'shoulder': 12,
            'elbow': 14,
            'wrist': 16,
            'hip': 24,
            'knee': 26,
            'ankle': 28,
            'foot': 32
        }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        self.feedback_count = len(self.settings['FEEDBACK_ID_MAP'])

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            'state_seq': [],

            'start_inactive_time_side': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME_SIDE': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,

            'DISPLAY_TEXT': np.full((self.feedback_count,), False),
            'COUNT_FRAMES': np.zeros((self.feedback_count,), dtype=np.int64),

            'INCORRECT_POSTURE': False,

            'prev_state': None,
            'curr_state': None,

            'CORRECT_COUNT': 0,
            'INCORRECT_COUNT': 0

        }

    def _update_state_sequence(self, current_state):

        if current_state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')) == 0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2') == 1)):
                self.state_tracker['state_seq'].append(current_state)

        elif current_state == 's3':
            if (current_state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']:
                self.state_tracker['state_seq'].append(current_state)

    def _show_feedback(self, frame, c_frame, dict_maps):

        for idx in np.where(c_frame)[0]:
            draw_text(
                frame,
                dict_maps[idx]['msg'],
                pos=dict_maps[idx]['pos'],
                text_color=dict_maps[idx]['text_color'],
                font_scale=0.6,
                text_color_bg=dict_maps[idx]['text_color_bg']
            )

        return frame

    def play_sound(self, path):
        audio_dir = f'./asset/audio/{path}.mp3'
        mixer.init()
        mixer.music.load(audio_dir)
        mixer.music.play()
        return
    
    def process_sit_up_with_weights(self, frame: np.array, pose):
        play_sound = None

        new_frame_time = time.time()
        fps = 1/(new_frame_time-self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps = str(int(fps))

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            landmark = keypoints.pose_landmarks.landmark

            # ------------------- Start Change Here 1 --------------
            nose_coord = get_landmark_features(
                landmark, self.dict_features, 'nose', frame_width, frame_height)

            _, left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'left', frame_width, frame_height)

            _, right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(
                left_shldr_coord, right_shldr_coord, nose_coord)
            # ------------------- End Change Here 1 --------------
            # ------------------- Start Change Here 2--------------
            if offset_angle > self.settings['OFFSET_THRESH']:
                # ------------------- End Change Here 2 --------------
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                    self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.settings['INACTIVE_THRESH']:
                    self.state_tracker['CORRECT_COUNT'] = 0
                    self.state_tracker['INCORRECT_COUNT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7,
                           self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7,
                           self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
                    frame,
                    'OFFSET ANGLE: '+str(offset_angle),
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                )

                # ------------------- Start Change Here 3--------------
                dist_left = abs(left_ankle_coord[1] - left_shldr_coord[1])
                dist_right = abs(right_ankle_coord[1] - right_shldr_coord[1])

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None

                if dist_left > dist_right:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    #foot_coord = left_foot_coord

                    multiplier = -1

                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    #foot_coord = right_foot_coord

                    multiplier = 1

                # ------------------- Verical Angle calculation --------------

                hip_vertical_angle = find_angle(
                    np.array([hip_coord[0], 0]), shldr_coord, hip_coord)
                if hip_vertical_angle <= 80:
                    cv2.ellipse(frame, hip_coord, (20, 20),
                                angle=0, startAngle=-30, endAngle=-20 + multiplier*hip_vertical_angle,
                                color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                draw_dotted_line(
                    frame, hip_coord, start=hip_coord[1]-50, end=hip_coord[1], line_color=self.COLORS['blue'])

                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                #cv2.line(frame, foot_coord, ankle_coord,
                #         self.COLORS['light_blue'], 4,  lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                #cv2.circle(frame, foot_coord, 7,
                #           self.COLORS['yellow'], -1,  lineType=self.linetype)

                current_state = None

                if self.settings['REF_ANGLE']['NORMAL'][0] <= int(hip_vertical_angle) <= self.settings['REF_ANGLE']['NORMAL'][1]:
                    current_state = 's1'
                elif self.settings['REF_ANGLE']['TRANS'][0] <= int(hip_vertical_angle) <= self.settings['REF_ANGLE']['TRANS'][1]:
                    current_state = 's2'
                elif self.settings['REF_ANGLE']['PASS'][0] <= int(hip_vertical_angle) <= self.settings['REF_ANGLE']['PASS'][1]:
                    current_state = 's3'

                # ------------------- End Change Here 3 --------------
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                hip_upper_y = hip_coord[1]-20
                ankle_lower_y = ankle_coord[1] + 20

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------
                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['CORRECT_COUNT'] += 1
                        play_sound = str(self.state_tracker['CORRECT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        pass

                    elif self.state_tracker['INCORRECT_POSTURE'] and len(self.state_tracker['state_seq']) != 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    # ------------------- Start Change Here 4--------------
                    
                    #if self.settings['HIP_THRESH'] > hip_vertical_angle:
                    #    self.state_tracker['DISPLAY_TEXT'][0] = True
                    #    self.state_tracker['INCORRECT_POSTURE'] = True

                    if hip_upper_y > ankle_lower_y:
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        self.play_sound('Situp_1')
                    # ------------------- End Change Here 4 --------------
                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                        self.state_tracker['start_inactive_time_side']
                    self.state_tracker['start_inactive_time_side'] = end_time

                    if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                        self.state_tracker['CORRECT_COUNT'] = 0
                        self.state_tracker['INCORRECT_COUNT'] = 0
                        display_inactivity = True

                else:

                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

                # -------------------------------------------------------------------------------------------------------
                # ------------------- Start Change Here 5 --------------
                hip_text_coord_x = hip_coord[0] + 20
                # ------------------- End Change Here 5 --------------
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 20

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
                frame = self._show_feedback(
                    frame, self.state_tracker['COUNT_FRAMES'], self.settings['FEEDBACK_ID_MAP'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                # ------------------- Start Change Here 6 --------------
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x,
                            hip_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                # ------------------- End Change Here 6 --------------
                if self.state_tracker['curr_state'] is not None:
                    draw_text(
                        frame,
                        "STAGE: " +
                        str(self.state_tracker['curr_state']).replace('s', ''),
                        pos=(int(frame_width*0.05), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.8,
                        text_color_bg=(128, 128, 128)
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    "FPS: " + fps,
                    pos=(int(frame_width*0.81), frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(102, 0, 204),

                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state

        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                self.state_tracker['start_inactive_time_side']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                self.state_tracker['CORRECT_COUNT'] = 0
                self.state_tracker['INCORRECT_COUNT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time_side'] = end_time

            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full(
                (self.feedback_count,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros(
                (self.feedback_count,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound

    def process_dumbbell_fly(self, frame: np.array, pose):
        play_sound = None

        new_frame_time = time.time()
        fps = 1/(new_frame_time-self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps = str(int(fps))

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            landmark = keypoints.pose_landmarks.landmark
            nose_coord = get_landmark_features(
                landmark, self.dict_features, 'nose', frame_width, frame_height)
            _, left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, _, left_ankle_coord, left_foot_coord = \
                get_landmark_features(
                    landmark, self.dict_features, 'left', frame_width, frame_height)
            _, right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, _, right_ankle_coord, right_foot_coord = \
                get_landmark_features(
                    landmark, self.dict_features, 'right', frame_width, frame_height)
            
            mid_shldr_coord = np.add(left_shldr_coord, right_shldr_coord) // 2
            
            #ankle_left_visibility = get_visibility(landmark, self.dict_features, 'left', 'ankle')
            #ankle_right_visibility = get_visibility(landmark, self.dict_features, 'right', 'ankle')

            foot_left_visibility = get_visibility(landmark, self.dict_features, 'left', 'foot')
            foot_right_visibility = get_visibility(landmark, self.dict_features, 'right', 'foot')

            hip_left_visibility = get_visibility(landmark, self.dict_features, 'left', 'hip')
            hip_right_visibility = get_visibility(landmark, self.dict_features, 'right', 'hip')

            #ankle_standing_prob = (ankle_left_visibility+ankle_right_visibility) / 2

            if (foot_left_visibility == 0) and (foot_right_visibility == 0):
                foot_standing_prob = 0
            else:
                foot_standing_prob = (foot_left_visibility+foot_right_visibility) / 2
            
            if (hip_left_visibility == 0) and (hip_right_visibility == 0):
                hip_standing_prob = 0
            else:
                hip_standing_prob = (hip_left_visibility+hip_right_visibility) / 2

            if (foot_standing_prob == 0) and (hip_standing_prob == 0):
                standing_prob = 0
            else:
                standing_prob = (foot_standing_prob + hip_standing_prob) / 2

            if ((foot_standing_prob > self.settings['OFFSET_THRESH']) or (hip_standing_prob > self.settings['OFFSET_THRESH'])) and \
                (mid_shldr_coord[1] < self.settings['SHLDR_THRESH']):
                
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                    self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.settings['INACTIVE_THRESH']:
                    self.state_tracker['CORRECT_COUNT'] = 0
                    self.state_tracker['INCORRECT_COUNT'] = 0
                    display_inactivity = True

                cv2.circle(frame, left_foot_coord, 7,
                           self.COLORS['magenta'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_foot_coord, 7,
                           self.COLORS['magenta'], -1,  lineType=self.linetype)
                cv2.line(frame, left_foot_coord, right_foot_coord,
                         self.COLORS['magenta'], 4,  lineType=self.linetype)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    'PLEASE LAY DOWN!!!', #'CAMERA NOT ALIGNED PROPERLY!!!'
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
                    frame,
                    'STANDING PROB: '+str(standing_prob),
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            else:
                
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                )

                right_multiplier = 1
                left_multiplier = -1

                # ------------------- Verical Angle calculation --------------

                left_shldr_vertical_angle = find_angle(
                    left_elbow_coord, np.array([left_shldr_coord[0]+0.1, 0]), left_shldr_coord)
                cv2.ellipse(frame, left_shldr_coord, (20, 20),
                            angle=0, startAngle=90, endAngle=90-left_multiplier*left_shldr_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                right_shldr_vertical_angle = find_angle(
                    right_elbow_coord, np.array([right_shldr_coord[0]+0.1, 0]), right_shldr_coord)
                cv2.ellipse(frame, right_shldr_coord, (20, 20),
                            angle=0, startAngle=90, endAngle=90-right_multiplier*right_shldr_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                left_shldr_wrist_elbow_angle = find_angle(
                    left_shldr_coord, left_wrist_coord, left_elbow_coord)
                cv2.ellipse(frame, left_elbow_coord, (20, 20),
                            angle=0, startAngle=90, endAngle=90-left_multiplier*left_shldr_wrist_elbow_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                right_shldr_wrist_elbow_angle = find_angle(
                    right_shldr_coord, right_wrist_coord, right_elbow_coord)
                cv2.ellipse(frame, right_elbow_coord, (20, 20),
                            angle=0, startAngle=90, endAngle=90-left_multiplier*right_shldr_wrist_elbow_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, left_shldr_coord, right_shldr_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, left_shldr_coord, left_elbow_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, left_elbow_coord, left_wrist_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, right_shldr_coord, right_elbow_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, right_elbow_coord, right_wrist_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, left_shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                #cv2.circle(frame, left_hip_coord, 7,
                #           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                #cv2.circle(frame, right_hip_coord, 7,
                #           self.COLORS['yellow'], -1,  lineType=self.linetype)
                #cv2.circle(frame, mid_knee_coord, 7,
                #           self.COLORS['red'], -1,  lineType=self.linetype)

                current_state = None

                diff_left_right_angle = abs(
                    left_shldr_vertical_angle - right_shldr_vertical_angle)

                if (diff_left_right_angle >= self.settings['DIFF_ANGLE_THRESH']):
                    self.state_tracker['DISPLAY_TEXT'][0] = True
                if self.settings['REF_ANGLE']['NORMAL'][0] <= int(left_shldr_vertical_angle) <= self.settings['REF_ANGLE']['NORMAL'][1] and diff_left_right_angle <= self.settings['DIFF_ANGLE_THRESH']:
                    current_state = 's1'
                elif self.settings['REF_ANGLE']['TRANS'][0] <= int(left_shldr_vertical_angle) <= self.settings['REF_ANGLE']['TRANS'][1] and diff_left_right_angle <= self.settings['DIFF_ANGLE_THRESH']:
                    current_state = 's2'
                elif self.settings['REF_ANGLE']['PASS'][0] <= int(left_shldr_vertical_angle) <= self.settings['REF_ANGLE']['PASS'][1] and diff_left_right_angle <= self.settings['DIFF_ANGLE_THRESH']:
                    current_state = 's3'
                 # ------------------- End Change Here 3 --------------
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------
                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['CORRECT_COUNT'] += 1
                        play_sound = str(self.state_tracker['CORRECT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE'] and len(self.state_tracker['state_seq']) != 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    # ------------------- Start Change Here 4--------------
                    if (left_shldr_wrist_elbow_angle < self.settings['ELBOW_THRESH'] or right_shldr_wrist_elbow_angle < self.settings['ELBOW_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        self.play_sound('Dumbbellfly_1')
                    # ------------------- End Change Here 4 --------------

                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                        self.state_tracker['start_inactive_time_side']
                    self.state_tracker['start_inactive_time_side'] = end_time

                    if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                        self.state_tracker['CORRECT_COUNT'] = 0
                        self.state_tracker['INCORRECT_COUNT'] = 0
                        display_inactivity = True

                else:

                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

                # -------------------------------------------------------------------------------------------------------
                # ------------------- Start Change Here 5 --------------
                left_shldr_text_coord_x = left_shldr_coord[0] - 15
                left_shldr_text_coord_y = left_shldr_coord[1] - 10
                right_shldr_text_coord_x = right_shldr_coord[0] - 15
                right_shldr_text_coord_x = right_shldr_coord[1] - 10
                left_elbow_text_coord_x = left_elbow_coord[0] - 15
                left_elbow_text_coord_y = left_elbow_coord[0] - 10
                right_elbow_text_coord_x = right_elbow_coord[0] - 15
                right_elbow_text_coord_y = right_elbow_coord[0] - 10

                # ------------------- End Change Here 5 --------------
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    left_shldr_text_coord_x = frame_width - \
                        left_shldr_coord[0] - 15
                    left_shldr_text_coord_y = left_shldr_coord[1] - 10
                    right_shldr_text_coord_x = frame_width - \
                        right_shldr_coord[0] - 15
                    right_shldr_text_coord_y = right_shldr_coord[1] - 10

                    left_elbow_text_coord_x = frame_width - \
                        left_elbow_coord[0] - 15
                    left_elbow_text_coord_y = left_shldr_coord[1] - 10
                    right_elbow_text_coord_x = frame_width - \
                        right_elbow_coord[0] - 15
                    right_elbow_text_coord_y = left_shldr_coord[1] - 10

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
                frame = self._show_feedback(
                    frame, self.state_tracker['COUNT_FRAMES'], self.settings['FEEDBACK_ID_MAP'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                # ------------------- Start Change Here 6 --------------
                cv2.putText(frame, str(int(left_shldr_vertical_angle)), (left_shldr_text_coord_x,
                            left_shldr_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_shldr_vertical_angle)), (right_shldr_text_coord_x,
                            right_shldr_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(left_shldr_wrist_elbow_angle)), (left_elbow_text_coord_x,
                            left_elbow_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_shldr_wrist_elbow_angle)), (right_elbow_text_coord_x,
                            right_elbow_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                                
                # ------------------- End Change Here 6 --------------
                if self.state_tracker['curr_state'] is not None:
                    draw_text(
                        frame,
                        "STAGE: " +
                        str(self.state_tracker['curr_state']).replace('s', ''),
                        pos=(int(frame_width*0.05), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.8,
                        text_color_bg=(128, 128, 128)
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    "FPS: " + fps,
                    pos=(int(frame_width*0.81), frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(102, 0, 204),

                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state
        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                self.state_tracker['start_inactive_time_side']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                self.state_tracker['CORRECT_COUNT'] = 0
                self.state_tracker['INCORRECT_COUNT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time_side'] = end_time

            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full(
                (self.feedback_count,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros(
                (self.feedback_count,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()
        return frame, play_sound

    def process_barbell_curl(self, frame: np.array, pose):
        play_sound = None

        new_frame_time = time.time()
        fps = 1/(new_frame_time-self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps = str(int(fps))

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            landmark = keypoints.pose_landmarks.landmark

            # ------------------- Start Change Here 1 --------------
            nose_coord = get_landmark_features(
                landmark, self.dict_features, 'nose', frame_width, frame_height)

            _, left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, _, _, left_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'left', frame_width, frame_height)

            _, right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, _, _, right_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(
                left_shldr_coord, right_shldr_coord, nose_coord)
            # ------------------- End Change Here 1 --------------
            # ------------------- Start Change Here 2--------------
            if offset_angle > self.settings['OFFSET_THRESH']:
                # ------------------- End Change Here 2 --------------
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                    self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.settings['INACTIVE_THRESH']:
                    self.state_tracker['CORRECT_COUNT'] = 0
                    self.state_tracker['INCORRECT_COUNT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7,
                           self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7,
                           self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
                    frame,
                    'OFFSET ANGLE: '+str(offset_angle),
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                )

                # ------------------- Start Change Here 3--------------
                dist_left = abs(left_hip_coord[1] - left_shldr_coord[1])
                dist_right = abs(right_hip_coord[1] - right_shldr_coord[1])

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None

                if dist_left > dist_right:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord

                    multiplier = -1

                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord

                    multiplier = 1

                # ------------------- Verical Angle calculation --------------

                wrist_shldr_elbow_angle = find_angle(
                    wrist_coord, shldr_coord, elbow_coord)
                cv2.ellipse(frame, elbow_coord, (15, 15),
                            angle=0, startAngle=-120, endAngle=-90-multiplier*wrist_shldr_elbow_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                hip_elbow_shldr_angle = find_angle(
                    hip_coord, elbow_coord, shldr_coord)
                cv2.ellipse(frame, hip_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier*hip_elbow_shldr_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                hip_vertical_angle = find_angle(
                    shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, shldr_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier*hip_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                draw_dotted_line(
                    frame, hip_coord, start=hip_coord[1]-50, end=hip_coord[1], line_color=self.COLORS['blue'])

                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)

                current_state = None

                if self.settings['REF_ANGLE']['NORMAL'][0] <= int(wrist_shldr_elbow_angle) <= self.settings['REF_ANGLE']['NORMAL'][1]:
                    current_state = 's1'
                elif self.settings['REF_ANGLE']['TRANS'][0] <= int(wrist_shldr_elbow_angle) <= self.settings['REF_ANGLE']['TRANS'][1]:
                    current_state = 's2'
                elif self.settings['REF_ANGLE']['PASS'][0] <= int(wrist_shldr_elbow_angle) <= self.settings['REF_ANGLE']['PASS'][1]:
                    current_state = 's3'

                # ------------------- End Change Here 3 --------------
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------
                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['CORRECT_COUNT'] += 1
                        play_sound = str(self.state_tracker['CORRECT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE'] and len(self.state_tracker['state_seq']) != 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    # ------------------- Start Change Here 4--------------
                    ply0 = False
                    ply1 = False


                    if self.settings['SHOULDER_THRESH'] < hip_elbow_shldr_angle:
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        ply0 = True

                    if self.settings['HIP_THRESH'] < hip_vertical_angle:
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        ply1 = True

                    if (ply0 == True and ply1 == True) or (ply0 == True and ply1 ==False) :
                        self.play_sound('Barbellcurl_0')
                    elif ply0 == False and ply1 == True:
                        self.play_sound('Barbellcurl_1')

                # ------------------- End Change Here 4 --------------
                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                        self.state_tracker['start_inactive_time_side']
                    self.state_tracker['start_inactive_time_side'] = end_time

                    if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                        self.state_tracker['CORRECT_COUNT'] = 0
                        self.state_tracker['INCORRECT_COUNT'] = 0
                        display_inactivity = True

                else:

                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

                # -------------------------------------------------------------------------------------------------------
                # ------------------- Start Change Here 5 --------------
                shldr_text_coord_x = shldr_coord[0] + 15
                elbow_text_coord_x = elbow_coord[0] + 10
                hip_text_coord_x = hip_coord[0] + 15
                # ------------------- End Change Here 5 --------------
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    elbow_text_coord_x = frame_width - elbow_coord[0] + 55
                    hip_text_coord_x = frame_width - hip_coord[0] + 20
                    shldr_text_coord_x = frame_width - shldr_coord[0] + 20

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
                frame = self._show_feedback(
                    frame, self.state_tracker['COUNT_FRAMES'], self.settings['FEEDBACK_ID_MAP'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                # ------------------- Start Change Here 6 --------------
                cv2.putText(frame, str(int(wrist_shldr_elbow_angle)), (elbow_text_coord_x,
                                                                       elbow_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(hip_elbow_shldr_angle)), (shldr_text_coord_x,
                                                                     shldr_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x,
                                                                  hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                # ------------------- End Change Here 6 --------------
                if self.state_tracker['curr_state'] is not None:
                    draw_text(
                        frame,
                        "STAGE: " +
                        str(self.state_tracker['curr_state']).replace('s', ''),
                        pos=(int(frame_width*0.05), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.8,
                        text_color_bg=(128, 128, 128)
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    "FPS: " + fps,
                    pos=(int(frame_width*0.81), frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(102, 0, 204),

                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state

        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                self.state_tracker['start_inactive_time_side']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                self.state_tracker['CORRECT_COUNT'] = 0
                self.state_tracker['INCORRECT_COUNT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time_side'] = end_time

            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full(
                (self.feedback_count,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros(
                (self.feedback_count,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound

    def process_dumbbell_lateral_raise(self, frame: np.array, pose):
        play_sound = None

        new_frame_time = time.time()
        fps = 1/(new_frame_time-self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps = str(int(fps))

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            landmark = keypoints.pose_landmarks.landmark
            nose_coord = get_landmark_features(
                landmark, self.dict_features, 'nose', frame_width, frame_height)
            _, left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, _, _, _ = \
                get_landmark_features(
                    landmark, self.dict_features, 'left', frame_width, frame_height)
            _, right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, _, _, _ = \
                get_landmark_features(
                    landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(
                left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle < self.settings['OFFSET_THRESH']:
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                    self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.settings['INACTIVE_THRESH']:
                    self.state_tracker['CORRECT_COUNT'] = 0
                    self.state_tracker['INCORRECT_COUNT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7,
                           self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7,
                           self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
                    frame,
                    'OFFSET ANGLE: '+str(offset_angle),
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            else:
                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                )

                right_multiplier = 1
                left_multiplier = -1

                # ------------------- Verical Angle calculation --------------

                left_hip_wrist_shldr_angle = find_angle(
                    left_hip_coord, left_wrist_coord, left_shldr_coord)
                cv2.ellipse(frame, left_shldr_coord, (20, 20),
                            angle=0, startAngle=90, endAngle=90-left_multiplier*left_hip_wrist_shldr_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                right_hip_wrist_shldr_angle = find_angle(
                    right_hip_coord, right_wrist_coord, right_shldr_coord)
                cv2.ellipse(frame, right_shldr_coord, (20, 20),
                            angle=0, startAngle=90, endAngle=90-right_multiplier*right_hip_wrist_shldr_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                left_shldr_wrist_elbow_angle = find_angle(
                    left_shldr_coord, left_wrist_coord, left_elbow_coord)
                cv2.ellipse(frame, left_elbow_coord, (20, 20),
                            angle=0, startAngle=90, endAngle=90-left_multiplier*left_shldr_wrist_elbow_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                right_shldr_wrist_elbow_angle = find_angle(
                    right_shldr_coord, right_wrist_coord, right_elbow_coord)
                cv2.ellipse(frame, right_elbow_coord, (20, 20),
                            angle=0, startAngle=90, endAngle=90-left_multiplier*right_shldr_wrist_elbow_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, left_shldr_coord, right_shldr_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, left_shldr_coord, left_elbow_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, left_elbow_coord, left_wrist_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, right_shldr_coord, right_elbow_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, right_elbow_coord, right_wrist_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, right_hip_coord, left_hip_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, right_hip_coord, right_shldr_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, left_hip_coord, left_shldr_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, left_shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_hip_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, left_wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_hip_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, right_wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)

                current_state = None

                diff_left_right_angle = abs(
                    left_hip_wrist_shldr_angle - right_hip_wrist_shldr_angle)

                if (diff_left_right_angle >= self.settings['DIFF_ANGLE_THRESH']):
                    self.state_tracker['DISPLAY_TEXT'][0] = True

                if self.settings['REF_ANGLE']['NORMAL'][0] <= int(left_hip_wrist_shldr_angle) <= self.settings['REF_ANGLE']['NORMAL'][1] and diff_left_right_angle <= self.settings['DIFF_ANGLE_THRESH']:
                    current_state = 's1'
                elif self.settings['REF_ANGLE']['TRANS'][0] <= int(left_hip_wrist_shldr_angle) <= self.settings['REF_ANGLE']['TRANS'][1] and diff_left_right_angle <= self.settings['DIFF_ANGLE_THRESH']:
                    current_state = 's2'
                elif self.settings['REF_ANGLE']['PASS'][0] <= int(left_hip_wrist_shldr_angle) <= self.settings['REF_ANGLE']['PASS'][1] and diff_left_right_angle <= self.settings['DIFF_ANGLE_THRESH']:
                    current_state = 's3'
                # # ------------------- End Change Here 3 --------------
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------
                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['CORRECT_COUNT'] += 1
                        play_sound = str(self.state_tracker['CORRECT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    # ------------------- Start Change Here 4--------------
                    if (left_shldr_wrist_elbow_angle < self.settings['ELBOW_THRESH'] or right_shldr_wrist_elbow_angle < self.settings['ELBOW_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        self.play_sound('Lateral_1')
                    # ------------------- End Change Here 4 --------------

                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                        self.state_tracker['start_inactive_time_side']
                    self.state_tracker['start_inactive_time_side'] = end_time

                    if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                        self.state_tracker['CORRECT_COUNT'] = 0
                        self.state_tracker['INCORRECT_COUNT'] = 0
                        display_inactivity = True

                else:

                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

                # -------------------------------------------------------------------------------------------------------
                # ------------------- Start Change Here 5 --------------
                left_shldr_text_coord_x = left_shldr_coord[0] - 15
                left_shldr_text_coord_y = left_shldr_coord[1] - 10
                right_shldr_text_coord_x = right_shldr_coord[0] - 15
                right_shldr_text_coord_x = right_shldr_coord[1] - 10
                left_elbow_text_coord_x = left_elbow_coord[0] - 15
                left_elbow_text_coord_y = left_elbow_coord[0] - 10
                right_elbow_text_coord_x = right_elbow_coord[0] - 15
                right_elbow_text_coord_y = right_elbow_coord[0] - 10

                # ------------------- End Change Here 5 --------------
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    left_shldr_text_coord_x = frame_width - \
                        left_shldr_coord[0] - 15
                    left_shldr_text_coord_y = left_shldr_coord[1] - 10
                    right_shldr_text_coord_x = frame_width - \
                        right_shldr_coord[0] - 15
                    right_shldr_text_coord_y = right_shldr_coord[1] - 10

                    left_elbow_text_coord_x = frame_width - \
                        left_elbow_coord[0] - 15
                    left_elbow_text_coord_y = left_shldr_coord[1] - 10
                    right_elbow_text_coord_x = frame_width - \
                        right_elbow_coord[0] - 15
                    right_elbow_text_coord_y = left_shldr_coord[1] - 10

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
                frame = self._show_feedback(
                    frame, self.state_tracker['COUNT_FRAMES'], self.settings['FEEDBACK_ID_MAP'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                # ------------------- Start Change Here 6 --------------
                cv2.putText(frame, str(int(left_hip_wrist_shldr_angle)), (left_shldr_text_coord_x,
                            left_shldr_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_hip_wrist_shldr_angle)), (right_shldr_text_coord_x,
                            right_shldr_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(left_shldr_wrist_elbow_angle)), (left_elbow_text_coord_x,
                            left_elbow_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(right_shldr_wrist_elbow_angle)), (right_elbow_text_coord_x,
                            right_elbow_text_coord_y), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                # ------------------- End Change Here 6 --------------
                if self.state_tracker['curr_state'] is not None:
                    draw_text(
                        frame,
                        "STAGE: " +
                        str(self.state_tracker['curr_state']).replace('s', ''),
                        pos=(int(frame_width*0.05), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.8,
                        text_color_bg=(128, 128, 128)
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    "FPS: " + fps,
                    pos=(int(frame_width*0.81), frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(102, 0, 204),

                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state
        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                self.state_tracker['start_inactive_time_side']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                self.state_tracker['CORRECT_COUNT'] = 0
                self.state_tracker['INCORRECT_COUNT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time_side'] = end_time

            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full(
                (self.feedback_count,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros(
                (self.feedback_count,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()
        return frame, play_sound

    def process_seated_tricep_press(self, frame: np.array, pose):
        play_sound = None

        new_frame_time = time.time()
        fps = 1/(new_frame_time-self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps = str(int(fps))

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            landmark = keypoints.pose_landmarks.landmark

            # ------------------- Start Change Here 1 --------------
            nose_coord = get_landmark_features(
                landmark, self.dict_features, 'nose', frame_width, frame_height)

            _, left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'left', frame_width, frame_height)

            _, right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(
                left_shldr_coord, right_shldr_coord, nose_coord)
            # ------------------- End Change Here 1 --------------
            # ------------------- Start Change Here 2--------------
            if offset_angle > self.settings['OFFSET_THRESH']:
                # ------------------- End Change Here 2 --------------
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                    self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.settings['INACTIVE_THRESH']:
                    self.state_tracker['CORRECT_COUNT'] = 0
                    self.state_tracker['INCORRECT_COUNT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7,
                           self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7,
                           self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
                    frame,
                    'OFFSET ANGLE: '+str(offset_angle),
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                )

                # ------------------- Start Change Here 3--------------
                dist_left = abs(left_ankle_coord[1] - left_shldr_coord[1])
                dist_right = abs(right_ankle_coord[1] - right_shldr_coord[1])

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None

                if dist_left > dist_right:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord

                    multiplier = -1

                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord

                    multiplier = 1

                # ------------------- Verical Angle calculation --------------

                wrist_shldr_elbow_angle = find_angle(
                    wrist_coord, shldr_coord, elbow_coord)
                cv2.ellipse(frame, elbow_coord, (15, 15),
                            angle=0, startAngle=-130, endAngle=-90-multiplier*wrist_shldr_elbow_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                hip_vertical_angle = find_angle(
                    np.array([hip_coord[0], 0]), shldr_coord, hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier*hip_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                draw_dotted_line(
                    frame, hip_coord, start=hip_coord[1]-50, end=hip_coord[1], line_color=self.COLORS['blue'])
#
                shldr_vertical_angle = find_angle(
                    np.array([shldr_coord[0], 0]), elbow_coord, shldr_coord)
                cv2.ellipse(frame, shldr_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier*shldr_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                draw_dotted_line(
                    frame, shldr_coord, start=shldr_coord[1]-50, end=shldr_coord[1], line_color=self.COLORS['blue'])

                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)

                current_state = None

                if self.settings['REF_ANGLE']['NORMAL'][0] <= int(wrist_shldr_elbow_angle) <= self.settings['REF_ANGLE']['NORMAL'][1]:
                    current_state = 's1'
                elif self.settings['REF_ANGLE']['TRANS'][0] <= int(wrist_shldr_elbow_angle) <= self.settings['REF_ANGLE']['TRANS'][1]:
                    current_state = 's2'
                elif self.settings['REF_ANGLE']['PASS'][0] <= int(wrist_shldr_elbow_angle) <= self.settings['REF_ANGLE']['PASS'][1]:
                    current_state = 's3'

                # ------------------- End Change Here 3 --------------
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------
                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['CORRECT_COUNT'] += 1
                        play_sound = str(self.state_tracker['CORRECT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    # ------------------- Start Change Here 4--------------
                    ply0 = False
                    ply1 = False

                    if self.settings['SHLDR_THRESH'] < shldr_vertical_angle:
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                        ply0 = True

                    if (hip_vertical_angle > self.settings['HIP_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        ply1 = True

                    if (ply0 == True and ply1 == True) or (ply0 == True and ply1 ==False) :
                        self.play_sound('Tricep_0')
                    elif ply0 == False and ply1 == True:
                        self.play_sound('Tricep_1')

                    # ------------------- End Change Here 4 --------------
                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                        self.state_tracker['start_inactive_time_side']
                    self.state_tracker['start_inactive_time_side'] = end_time

                    if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                        self.state_tracker['CORRECT_COUNT'] = 0
                        self.state_tracker['INCORRECT_COUNT'] = 0
                        display_inactivity = True

                else:

                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

                # -------------------------------------------------------------------------------------------------------
                # ------------------- Start Change Here 5 --------------
                elbow_text_coord_x = elbow_coord[0] + 55
                hip_text_coord_x = hip_coord[0] + 20
                shldr_text_coord_x = shldr_coord[0] + 20
                # ------------------- End Change Here 5 --------------
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    elbow_text_coord_x = frame_width - elbow_coord[0] + 55
                    hip_text_coord_x = frame_width - hip_coord[0] + 20
                    shldr_text_coord_x = frame_width - shldr_coord[0] + 20

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
                frame = self._show_feedback(
                    frame, self.state_tracker['COUNT_FRAMES'], self.settings['FEEDBACK_ID_MAP'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                # ------------------- Start Change Here 6 --------------
                cv2.putText(frame, str(int(wrist_shldr_elbow_angle)), (elbow_text_coord_x,
                            elbow_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x,
                            hip_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(shldr_vertical_angle)), (shldr_text_coord_x,
                            shldr_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                # ------------------- End Change Here 6 --------------
                if self.state_tracker['curr_state'] is not None:
                    draw_text(
                        frame,
                        "STAGE: " +
                        str(self.state_tracker['curr_state']).replace('s', ''),
                        pos=(int(frame_width*0.05), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.8,
                        text_color_bg=(128, 128, 128)
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    "FPS: " + fps,
                    pos=(int(frame_width*0.81), frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(102, 0, 204),

                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state

        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                self.state_tracker['start_inactive_time_side']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                self.state_tracker['CORRECT_COUNT'] = 0
                self.state_tracker['INCORRECT_COUNT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time_side'] = end_time

            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full(
                (self.feedback_count,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros(
                (self.feedback_count,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound

    def process_bent_over_dumbbell_row(self, frame: np.array, pose):
        play_sound = None

        new_frame_time = time.time()
        fps = 1/(new_frame_time-self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps = str(int(fps))

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            landmark = keypoints.pose_landmarks.landmark

            # ------------------- Start Change Here 1 --------------
            nose_coord = get_landmark_features(
                landmark, self.dict_features, 'nose', frame_width, frame_height)

            left_ear_coord, left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'left', frame_width, frame_height)

            right_ear_coord, right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(
                left_shldr_coord, right_shldr_coord, nose_coord)
            # ------------------- End Change Here 1 --------------

            # ------------------- Start Change Here 2--------------
            if offset_angle > self.settings['OFFSET_THRESH']:
                # ------------------- End Change Here 2 --------------
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                    self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.settings['INACTIVE_THRESH']:
                    self.state_tracker['CORRECT_COUNT'] = 0
                    self.state_tracker['INCORRECT_COUNT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7,
                           self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7,
                           self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
                    frame,
                    'OFFSET ANGLE: '+str(offset_angle),
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                )

                # ------------------- Start Change Here 3--------------
                dist_left = abs(left_foot_coord[1] - left_hip_coord[1])
                dist_right = abs(right_foot_coord[1] - right_hip_coord[1])

                ear_coord = None
                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_left > dist_right:
                    ear_coord = left_ear_coord
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord

                    multiplier = -1

                else:
                    ear_coord = right_ear_coord
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord

                    multiplier = 1

                # ------------------- Verical Angle calculation --------------

                elbow_hip_shldr_angle = find_angle(
                    elbow_coord, hip_coord, shldr_coord)
                cv2.ellipse(frame, shldr_coord, (30, 30),
                            angle=0, startAngle=45, endAngle=45-multiplier*elbow_hip_shldr_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
#
                hip_vertical_angle = find_angle(
                    np.array([hip_coord[0], 0]), shldr_coord, hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier*hip_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                draw_dotted_line(
                    frame, hip_coord, start=hip_coord[1]-50, end=hip_coord[1], line_color=self.COLORS['blue'])
#
                ankle_vertical_angle = find_angle(
                    knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier*ankle_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)
                draw_dotted_line(
                    frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])
#
                ear_hip_shldr_angle = find_angle(
                    ear_coord, hip_coord, shldr_coord)

                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, ear_coord, shldr_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord,
                         self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, ear_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, shldr_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)

                current_state = None

                if self.settings['REF_ANGLE']['NORMAL'][0] <= int(elbow_hip_shldr_angle) <= self.settings['REF_ANGLE']['NORMAL'][1]:
                    current_state = 's1'
                elif self.settings['REF_ANGLE']['TRANS'][0] <= int(elbow_hip_shldr_angle) <= self.settings['REF_ANGLE']['TRANS'][1]:
                    current_state = 's2'
                elif self.settings['REF_ANGLE']['PASS'][0] <= int(elbow_hip_shldr_angle) <= self.settings['REF_ANGLE']['PASS'][1]:
                    current_state = 's3'
                # ------------------- End Change Here 3 --------------
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------
                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['CORRECT_COUNT'] += 1
                        play_sound = str(self.state_tracker['CORRECT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE'] and len(self.state_tracker['state_seq']) != 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    # ------------------- Start Change Here 4--------------
                    ply0 = False
                    ply1 = False
                    ply2 = False


                    if (hip_vertical_angle < self.settings['HIP_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        ply0 = True
#
                    if (ankle_vertical_angle > self.settings['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        ply1 = True

                    if (self.settings['SHLDR_THRESH'] > ear_hip_shldr_angle):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        ply2 = True

                    if ply1 == True:
                        self.play_sound('Bentover_1')
                    elif ply2 == True:
                        self.play_sound('Bentover_2')
                    elif ply0 == True:
                        self.play_sound('Bentover_0')

                # ------------------- End Change Here 4 --------------
                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                        self.state_tracker['start_inactive_time_side']
                    self.state_tracker['start_inactive_time_side'] = end_time

                    if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                        self.state_tracker['CORRECT_COUNT'] = 0
                        self.state_tracker['INCORRECT_COUNT'] = 0
                        display_inactivity = True

                else:

                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

                # -------------------------------------------------------------------------------------------------------
                # ------------------- Start Change Here 5 --------------
                shldr_text_coord_x = shldr_coord[0] + 15
                hip_text_coord_x = hip_coord[0] + 15

                # ------------------- End Change Here 5 --------------
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    shldr_text_coord_x = frame_width - shldr_coord[0] + 15
                    hip_text_coord_x = frame_width - hip_coord[0] + 15
                
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
                frame = self._show_feedback(
                    frame, self.state_tracker['COUNT_FRAMES'], self.settings['FEEDBACK_ID_MAP'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                # ------------------- Start Change Here 6 --------------
                cv2.putText(frame, str(int(ear_hip_shldr_angle)), (shldr_text_coord_x,
                            shldr_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x,
                            hip_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                # ------------------- End Change Here 6 --------------
                if self.state_tracker['curr_state'] is not None:
                    draw_text(
                        frame,
                        "STAGE: " +
                        str(self.state_tracker['curr_state']).replace('s', ''),
                        pos=(int(frame_width*0.05), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.8,
                        text_color_bg=(128, 128, 128)
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    "FPS: " + fps,
                    pos=(int(frame_width*0.81), frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(102, 0, 204),

                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state

        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                self.state_tracker['start_inactive_time_side']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                self.state_tracker['CORRECT_COUNT'] = 0
                self.state_tracker['INCORRECT_COUNT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time_side'] = end_time

            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full(
                (self.feedback_count,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros(
                (self.feedback_count,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound

    def process_squat_with_weights(self, frame: np.array, pose):
        play_sound = None

        new_frame_time = time.time()
        fps = 1/(new_frame_time-self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps = str(int(fps))

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            landmark = keypoints.pose_landmarks.landmark

            # ------------------- Start Change Here 1 --------------
            nose_coord = get_landmark_features(
                landmark, self.dict_features, 'nose', frame_width, frame_height)

            _, left_shldr_coord, _, _, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'left', frame_width, frame_height)

            _, right_shldr_coord, _, _, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = get_landmark_features(
                landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(
                left_shldr_coord, right_shldr_coord, nose_coord)
            # ------------------- End Change Here 1 --------------

            # ------------------- Start Change Here 2--------------
            if offset_angle > self.settings['OFFSET_THRESH']:
                # ------------------- End Change Here 2 --------------
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - \
                    self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.settings['INACTIVE_THRESH']:
                    self.state_tracker['CORRECT_COUNT'] = 0
                    self.state_tracker['INCORRECT_COUNT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7,
                           self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7,
                           self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    'CAMERA NOT ALIGNED PROPERLY!!!',
                    pos=(30, frame_height-60),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                draw_text(
                    frame,
                    'OFFSET ANGLE: '+str(offset_angle),
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                )

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                self.state_tracker['prev_state'] = None
                self.state_tracker['curr_state'] = None

            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter(
                )

                # ------------------- Start Change Here 3--------------
                dist_left = abs(left_foot_coord[1] - left_hip_coord[1])
                dist_right = abs(right_foot_coord[1] - right_hip_coord[1])

                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_left > dist_right:
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord

                    multiplier = -1

                else:
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord

                    multiplier = 1

                # ------------------- Verical Angle calculation --------------

                knee_vertical_angle = find_angle(
                    hip_coord, np.array([knee_coord[0]+0.1, 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20),
                            angle=0, startAngle=-90, endAngle=-90-multiplier*knee_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                draw_dotted_line(
                    frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])

                ankle_vertical_angle = find_angle(
                    knee_coord, np.array([ankle_coord[0]+0.1, 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle=0, startAngle=-90, endAngle=-90 + multiplier*ankle_vertical_angle,
                            color=self.COLORS['white'], thickness=3,  lineType=self.linetype)

                draw_dotted_line(
                    frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])
                # ------------------------------------------------------------

                # Join landmarks.
                cv2.line(frame, knee_coord, hip_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord,
                         self.COLORS['light_blue'], 4,  lineType=self.linetype)

                # Plot landmark points
                cv2.circle(frame, hip_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7,
                           self.COLORS['yellow'], -1,  lineType=self.linetype)

                current_state = None

                if self.settings['REF_ANGLE']['NORMAL'][0] <= int(knee_vertical_angle) <= self.settings['REF_ANGLE']['NORMAL'][1]:
                    current_state = 's1'
                elif self.settings['REF_ANGLE']['TRANS'][0] <= int(knee_vertical_angle) <= self.settings['REF_ANGLE']['TRANS'][1]:
                    current_state = 's2'
                elif self.settings['REF_ANGLE']['PASS'][0] <= int(knee_vertical_angle) <= self.settings['REF_ANGLE']['PASS'][1]:
                    current_state = 's3'

                # ------------------- End Change Here 3 --------------
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)

                # -------------------------------------- COMPUTE COUNTERS --------------------------------------
                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['CORRECT_COUNT'] += 1
                        play_sound = str(self.state_tracker['CORRECT_COUNT'])

                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq']) == 1:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['INCORRECT_COUNT'] += 1
                        play_sound = 'incorrect'

                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                # ----------------------------------------------------------------------------------------------------

                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    # ------------------- Start Change Here 4--------------
                    ply2 = False
                    ply1 = False

                    if self.settings['KNEE_THRESH'][0] < knee_vertical_angle < self.settings['KNEE_THRESH'][1] and \
                       self.state_tracker['state_seq'].count('s2') == 1 :
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                        
                    elif knee_vertical_angle > self.settings['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        ply2 = True

                    elif (ankle_vertical_angle > self.settings['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][1] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True
                        ply1 = True
                    
                    if ply2 == True:
                        self.play_sound('Squat_2')
                    elif ply2 == False and ply1 == True:
                        self.play_sound('Squat_1')

                    # ------------------- End Change Here 4 --------------
                # ----------------------------------------------------------------------------------------------------

                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False

                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                        self.state_tracker['start_inactive_time_side']
                    self.state_tracker['start_inactive_time_side'] = end_time

                    if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                        self.state_tracker['CORRECT_COUNT'] = 0
                        self.state_tracker['INCORRECT_COUNT'] = 0
                        display_inactivity = True

                else:

                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

                # -------------------------------------------------------------------------------------------------------
                # ------------------- Start Change Here 5 --------------
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10
                # ------------------- End Change Here 5 --------------
                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']] += 1
                frame = self._show_feedback(
                    frame, self.state_tracker['COUNT_FRAMES'], self.settings['FEEDBACK_ID_MAP'])

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                    )
                    self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0
                # ------------------- Start Change Here 6 --------------
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x,
                            knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x,
                            ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                # ------------------- End Change Here 6 --------------
                if self.state_tracker['curr_state'] is not None:
                    draw_text(
                        frame,
                        "STAGE: " +
                        str(self.state_tracker['curr_state']).replace('s', ''),
                        pos=(int(frame_width*0.05), 30),
                        text_color=(255, 255, 230),
                        font_scale=0.8,
                        text_color_bg=(128, 128, 128)
                    )

                draw_text(
                    frame,
                    "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )

                draw_text(
                    frame,
                    "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),

                )

                draw_text(
                    frame,
                    "FPS: " + fps,
                    pos=(int(frame_width*0.81), frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(102, 0, 204),

                )

                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES']
                                                   > self.settings['CNT_FRAME_THRESH']] = 0
                self.state_tracker['prev_state'] = current_state

        else:
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME_SIDE'] += end_time - \
                self.state_tracker['start_inactive_time_side']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME_SIDE'] >= self.settings['INACTIVE_THRESH']:
                self.state_tracker['CORRECT_COUNT'] = 0
                self.state_tracker['INCORRECT_COUNT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time_side'] = end_time

            draw_text(
                frame,
                "CORRECT: " + str(self.state_tracker['CORRECT_COUNT']),
                pos=(int(frame_width*0.68), 30),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(18, 185, 0)
            )

            draw_text(
                frame,
                "INCORRECT: " + str(self.state_tracker['INCORRECT_COUNT']),
                pos=(int(frame_width*0.68), 80),
                text_color=(255, 255, 230),
                font_scale=0.7,
                text_color_bg=(221, 0, 0),

            )

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time_side'] = time.perf_counter(
                )
                self.state_tracker['INACTIVE_TIME_SIDE'] = 0.0

            # Reset all other state variables

            self.state_tracker['prev_state'] = None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full(
                (self.feedback_count,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros(
                (self.feedback_count,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

        return frame, play_sound
