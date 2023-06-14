def get_sit_up_with_weights():

    _ANGLE_HIP_SHLDR_VERT = {
        'NORMAL': (80,  90),
        'TRANS': (36, 79),
        'PASS': (20, 35)
    }

    settings = {

        'REF_ANGLE': _ANGLE_HIP_SHLDR_VERT,

        #'HIP_THRESH': 15,

        'OFFSET_THRESH': 40.0,
        'INACTIVE_THRESH': 10.0,

        'CNT_FRAME_THRESH': 50,

        'FEEDBACK_ID_MAP': {
            0: {'msg': 'Bending Over Too Much',
                'pos': (30, 80),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
            1: {'msg': 'Keep your feet on the floor',
                'pos': (30, 130),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
        }

    }

    return settings


def get_dumbbell_fly():

    _ANGLE_ELBOW_HIP_SHLDR = {
        'NORMAL': (1,  30),
        'TRANS': (31, 74),
        'PASS': (75, 120)
    }

    settings = {

        'REF_ANGLE': _ANGLE_ELBOW_HIP_SHLDR,

        'DIFF_ANGLE_THRESH': 10.0,

        'ELBOW_THRESH': 90,
        'SHLDR_THRESH': 200,

        'OFFSET_THRESH': 0.3,
        'INACTIVE_THRESH': 10.0,

        'CNT_FRAME_THRESH': 50,

        'FEEDBACK_ID_MAP': {
            0: {'msg': 'LEFT & RIGHT NOT BALANCE',
                'pos': (30, 80),
                'text_color': (0, 0, 0),
                'text_color_bg': (255, 255, 0)},
            1: {'msg': 'REACH YOUR ARM',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
        }

    }

    return settings


def get_barbell_curl():

    _ANGLE_WRIST_ELBOW_VERT = {
        'NORMAL': (110, 145),
        'TRANS': (75, 109),
        'PASS': (10, 50)
    }

    settings = {

        'REF_ANGLE': _ANGLE_WRIST_ELBOW_VERT,

        'SHOULDER_THRESH': 25,
        #'ELBOW_THRESH': 145,
        'HIP_THRESH': 10,

        'OFFSET_THRESH': 80.0,
        'INACTIVE_THRESH': 10.0,

        'CNT_FRAME_THRESH': 50,

        'FEEDBACK_ID_MAP': {
            0: {'msg': 'STRAIGHT YOUR SHOULDER',
                'pos': (30, 80),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
            1: {'msg': 'STRAIGHT YOUR BACK',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
            2: {'msg': 'SWING TOO MUCH',
                'pos': (30, 125),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)}
        }

    }

    return settings


def get_dumbbell_lateral_raise():

    _ANGLE_ELBOW_HIP_SHLDR = {
        'NORMAL': (15,  30),
        'TRANS': (31, 89),
        'PASS': (90, 120)
    }

    settings = {

        'REF_ANGLE': _ANGLE_ELBOW_HIP_SHLDR,

        'DIFF_ANGLE_THRESH': 15.0,

        'ELBOW_THRESH': 90,

        'OFFSET_THRESH': 55.0,
        'INACTIVE_THRESH': 10.0,

        'CNT_FRAME_THRESH': 50,

        'FEEDBACK_ID_MAP': {
            0: {'msg': 'LEFT & RIGHT NOT BALANCE',
                'pos': (30, 80),
                'text_color': (0, 0, 0),
                'text_color_bg': (255, 255, 0)},
            1: {'msg': 'REACH YOUR ARM',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
        }

    }

    return settings


def get_seated_tricep_press():

    _ANGLE_WRIST_SHLDR_ELBOW = {
        'NORMAL': (155,  180),
        'TRANS': (130, 156),
        'PASS': (70, 131)
    }

    settings = {

        'REF_ANGLE': _ANGLE_WRIST_SHLDR_ELBOW,

        'SHLDR_THRESH': 30,
        'HIP_THRESH': 15,

        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 10.0,

        'CNT_FRAME_THRESH': 50,

        'FEEDBACK_ID_MAP': {
            0: {'msg': 'Keep your arms near your ears',
                'pos': (30, 80),
                'text_color': (0, 0, 0),
                'text_color_bg': (255, 255, 0)},
            1: {'msg': 'Keep your body stright',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
        }

    }

    return settings


def get_bent_over_dumbbell_row():

    _ANGLE_ELBOW_HIP_SHLDR = {
        'NORMAL': (30,  55),
        'TRANS': (10, 29),
        'PASS': (0, 9)
    }

    settings = {

        'REF_ANGLE': _ANGLE_ELBOW_HIP_SHLDR,

        'SHLDR_THRESH': 145,
        'ANKLE_THRESH': 45,
        'HIP_THRESH': 40,

        'OFFSET_THRESH': 55.0,
        'INACTIVE_THRESH': 10.0,

        'CNT_FRAME_THRESH': 50,

        'FEEDBACK_ID_MAP': {
            0: {'msg': 'LOWER YOUR BACK',
                'pos': (30, 80),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
            1: {'msg': 'KNEE FALLING OVER TOE',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
            2: {'msg': 'STRAIGHTEN YOUR BACK',
                'pos': (30, 200),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)}
        }

    }

    return settings


def get_squat_with_weights():

    _ANGLE_HIP_KNEE_VERT = {
        'NORMAL': (0,  32),
        'TRANS': (35, 65),
        'PASS': (80, 95)
    }

    settings = {

        'REF_ANGLE': _ANGLE_HIP_KNEE_VERT,

        'HIP_THRESH': [10, 50],
        'ANKLE_THRESH': 45,
        'KNEE_THRESH': [70, 80, 90],

        'OFFSET_THRESH': 55.0,
        'INACTIVE_THRESH': 10.0,

        'CNT_FRAME_THRESH': 50,

        'FEEDBACK_ID_MAP': {
            0: {'msg': 'GO UP',
                'pos': (30, 80),
                'text_color': (0, 0, 0),
                'text_color_bg': (255, 255, 0)},
            1: {'msg': 'KNEE FALLING OVER TOE',
                'pos': (30, 170),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)},
            2: {'msg': 'SQUAT TOO DEEP',
                'pos': (30, 125),
                'text_color': (255, 255, 230),
                'text_color_bg': (255, 80, 80)}
        }

    }

    return settings
