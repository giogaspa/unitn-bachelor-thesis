import numpy as np

from common.utils import compute_vertical_angle
from common.params import image_height, image_width, movenet_image_landscape_height_offset

PROBABILITY_THRESHOLD = 0.3

def process(raw_data):
    print("Process data")
    # Keep only useful keypoints (eyes, ears, nose, shoulders) with good probability
    data = raw_data[:,:21]
    data = data.astype(np.float32)

    classes = raw_data[:,-1]

    processed_data = [[                    
        'nose-x', 'nose-y',
        'left_eye-x', 'left_eye-y',
        'right_eye-x', 'right_eye-y',
        #'left_ear-x', 'left_ear-y',
        #'right_ear-x', 'right_ear-y',
        'left_shoulder-x', 'left_shoulder-y',
        'right_shoulder-x', 'right_shoulder-y',
        'head_angle', 'shoulder_angle',
        'class'
    ]]

    # FEATURE SELECTION
    for idx, r in enumerate(data):
        if (
            r[2] >= PROBABILITY_THRESHOLD 
            and r[5] >= PROBABILITY_THRESHOLD
            and r[8] >= PROBABILITY_THRESHOLD
            #and r[11] >= PROBABILITY_THRESHOLD
            #and r[14] >= PROBABILITY_THRESHOLD
            and r[17] >= PROBABILITY_THRESHOLD
            and r[20] >= PROBABILITY_THRESHOLD
            ):
            # FEATURE AUGMENTATION -> aggiungo angolo spalle rispetto all'asse x e angolo testa rispetto all'asse y
            head_angle = compute_head_angle(r) / 90
            shoulder_angle = compute_shoulder_angle(r) / 90

            #processed_data.append([*r[0:2], *r[3:5], *r[6:8], *r[9:11], *r[12:14], *r[15:17], *r[18:20], head_angle, shoulder_angle, classes[idx]])
            processed_data.append([*r[0:2], *r[3:5], *r[6:8], *r[15:17], *r[18:20], head_angle, shoulder_angle, classes[idx]])

    processed_data = np.array(processed_data).reshape([-1, 13])

    # FEATURE SCALING -> not necessary beacause all data are between 0 and 1

    return processed_data

def compute_head_angle(row):
    left_eye = get_cordinates_from(row[3:6])
    right_eye = get_cordinates_from(row[6:9])

    return compute_vertical_angle(right_eye, left_eye)

def compute_shoulder_angle(row):
    left_shoulder = get_cordinates_from(row[15:18])
    right_shoulder = get_cordinates_from(row[18:21])

    return compute_vertical_angle(right_shoulder, left_shoulder)

def get_cordinates_from(normalized_coord):
    return [int(normalized_coord[0] * image_width), int(normalized_coord[1] * image_height) - movenet_image_landscape_height_offset, normalized_coord[2]]