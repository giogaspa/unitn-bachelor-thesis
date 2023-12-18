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

image_width = 640
image_height = 640
image_ar = image_height / image_width

movenet_image_height = 256
movenet_image_width = 256
movenet_image_landscape_height_offset = int((movenet_image_height - (image_ar * movenet_image_width)) / 2)

exercises = {
    "ex1" : {
        "name" : "Piegamenti laterali del capo",
        "filename" : "piegamenti-laterali-del-capo",
        "captureCommands" : [
            {
                "key" : 1,
                "className" : "center"
            },
            {
                "key" : 2,
                "className" : "left"
            },
            {
                "key" : 3,
                "className" : "right"
            }
        ]
    }
}

camera_list = [
    {
        "id" : "mac-webcam",
        "name" : "Mac webcam",
        "resolution" : (640,480),
        "notes" : ""
    },
    {
        "id" : "logitech",
        "name" : "Logitech webcam",
        "resolution" : (640,480),
        "notes" : "720p"
    }
]

# Posizione della camera rispetto al corpo
camera_position = [
    "frontal",
    "lateral-left",
    "lateral-right",
]

subject = [
    "gioacchino",
    "veronica"
]

subject_position = [
    "sitting",
    "upright",
    "lying"
]

acquisition_info = {
    "exercise_id": None,
    "exercise" : None,
    "camera_id": None,
    "camera_position": None,
    "subject": None,
    "subject_position" : None,
    "model": "movenet_thunder"
}


## KEYS 
# z = 122
# x = 120
# c = 99