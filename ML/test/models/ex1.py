import numpy as np
import tensorflow as tf

from common.params import exercises
from common.utils import compute_vertical_angle

exerciseId = 'ex1'
class_transformer = list(exercises[exerciseId]['transformer'].keys())
model = None

def load_model(model_filepath):
    global model

    ## Load model
    model = tf.keras.models.load_model(model_filepath)

def predict(kp_raw):
    global model, class_transformer

    head_angle = compute_vertical_angle([kp_raw['right_eye'][0][1],kp_raw['right_eye'][0][0]],[kp_raw['left_eye'][0][1],kp_raw['left_eye'][0][0]]) / 90
    shoulder_angle = compute_vertical_angle([kp_raw['right_shoulder'][0][1],kp_raw['right_shoulder'][0][0]], [kp_raw['left_shoulder'][0][1],kp_raw['left_shoulder'][0][0]]) / 90

    data = np.array([
            kp_raw['nose'][0][1],kp_raw['nose'][0][0],
            kp_raw['left_eye'][0][1],kp_raw['left_eye'][0][0],
            kp_raw['right_eye'][0][1],kp_raw['right_eye'][0][0],
            #kp_raw['left_ear'][0][1],kp_raw['left_ear'][0][0],
            #kp_raw['right_ear'][0][1],kp_raw['right_ear'][0][0],
            kp_raw['left_shoulder'][0][1],kp_raw['left_shoulder'][0][0],
            kp_raw['right_shoulder'][0][1],kp_raw['right_shoulder'][0][0],
            head_angle, shoulder_angle
    ])

    data = tf.constant(data.astype(np.float32).reshape([1, len(data)]))
    predicted = model.predict(data, verbose=0)
    predicted = tf.nn.softmax(predicted).numpy()
    predicted_class = class_transformer[np.argmax(predicted)]

    return predicted_class