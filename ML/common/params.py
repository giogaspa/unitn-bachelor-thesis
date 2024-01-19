import os
import pathlib


_dir = pathlib.Path(__file__).parent.resolve()

KEY_POINTS = {
    'nose' : 0 , 
    'left_eye' : 1, 
    'right_eye' : 2, 
    'left_ear' : 3, 
    'right_ear' : 4, 
    'left_shoulder' : 5, 
    'right_shoulder' : 6, 
    'left_elbow' : 7, 
    'right_elbow' : 8, 
    'left_wrist' : 9, 
    'right_wrist' : 10, 
    'left_hip' : 11, 
    'right_hip' : 12, 
    'left_knee' : 13, 
    'right_knee' : 14, 
    'left_ankle' : 15, 
    'right_ankle' : 16
}

KEY_POINTS_NAMES = list(KEY_POINTS.keys())

DATASET_PATH = os.path.join(_dir, '..', 'data')

exercises = {
    "ex1" : {
        "name" : "Piegamenti laterali del capo",
        "filename" : "piegamenti-laterali-del-capo",
        "captureCommands" : [
            {
                "key" : 1,   # arrow down
                "className" : "center"
            },
            {
                "key" : 2,   # arrow left
                "className" : "left"
            },
            {
                "key" : 113, # q
                "className" : "half-left"
            },
            {
                "key" : 97,  # a
                "className" : "wrong-left-forward"
            },
            {
                "key" : 122, # z
                "className" : "wrong-left-shoulder"
            },
            {
                "key" : 3,   # arrow right
                "className" : "right"
            },
            {
                "key" : 101, # e
                "className" : "half-right"
            },
            {
                "key" : 100, # d
                "className" : "wrong-right-forward"
            },
            {
                "key" : 99,  # c
                "className" : "wrong-right-shoulder"
            }
        ],
        "transformer": {
            "center": 0,
            "right": 1,
            "half-right": 2,
            "wrong-right-forward": 3,
            "wrong-right-shoulder": 4,
            "left": 5,
            "half-left": 6,
            "wrong-left-forward": 7,
            "wrong-left-shoulder": 8,
        }
    }
}

image_width = 640
image_height = 640
image_ar = image_height / image_width

movenet_image_height = 256
movenet_image_width = 256
movenet_image_landscape_height_offset = int((movenet_image_height - (image_ar * movenet_image_width)) / 2)