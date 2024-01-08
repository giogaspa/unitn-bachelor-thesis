import numpy as np


PROBABILITY_THRESHOLD = 0.2

def process(raw_data):
    print("Process data")
    # Keep only useful keypoints (eyes, ears, nose, shoulders) with good probability
    data = raw_data[:,:21]
    data = data.astype(np.float32)

    classes = raw_data[:,53]

    processed_data = [[                    
        'nose-x', 'nose-y',
        'left_eye-x', 'left_eye-y',
        'right_eye-x', 'right_eye-y',
        'left_ear-x', 'left_ear-y',
        'right_ear-x', 'right_ear-y',
        'left_shoulder-x', 'left_shoulder-y',
        'right_shoulder-x', 'right_shoulder-y',
        'class'
    ]]

    for idx, r in enumerate(data):
        if (
            r[2] >= PROBABILITY_THRESHOLD 
            and r[5] >= PROBABILITY_THRESHOLD
            and r[8] >= PROBABILITY_THRESHOLD
            and r[11] >= PROBABILITY_THRESHOLD
            and r[14] >= PROBABILITY_THRESHOLD
            and r[17] >= PROBABILITY_THRESHOLD
            and r[20] >= PROBABILITY_THRESHOLD
            ):
            processed_data.append([*r[0:2], *r[3:5], *r[6:8], *r[9:11], *r[12:14], *r[15:17], *r[18:20], classes[idx]])

    processed_data = np.array(processed_data).reshape([-1, 15]) # TODO test -1

    # TODO Data augmentation???

    return processed_data


