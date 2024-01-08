import sys
import os
import math
import pathlib
import csv
import cv2 as cv
import numpy as np
import tensorflow as tf
from rich import print
from rich.progress import Progress
from uuid import uuid4
from datetime import datetime

# Load common module path
_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(_dir, '..'))

from common.params import DATASET_PATH, KEY_POINTS, KEY_POINTS_NAMES
from common.file import make_folder, count_rows
from common.movenet import movenet
from params import *

export_data = []

def get_cordinates_from(normalized_coord):
    return [int(normalized_coord[1] * image_width), int(normalized_coord[0] * image_height) - movenet_image_landscape_height_offset, normalized_coord[2]]
    
def get_keypoint_axes_from_prediction(prediction, kp='nose'):
    coords = prediction[KEY_POINTS.get(kp)]
    return get_cordinates_from(coords[:3])

def compute_key_points(image):
    # Ridimensiona il frame e aggiungi un padding se necessario. Il ridimensionamento trasforma i dati in float32
    tImage = tf.image.resize_with_pad(image, movenet_image_height, movenet_image_width)

    # Converti image data type da float32 -> int32 
    tImage = tf.cast(tImage, dtype=tf.int32)

    # Aggiungi una dimensione (in testa) in modo da essere compatibile con la shape dei dati richiesta dal modello
    tImage = tf.expand_dims(tImage, axis=0)

    # Movenet si aspetta in input: A frame of video or an image, 
    # represented as an int32 tensor of shape: 192x192x3. 
    # Channels order: RGB with values in [0, 255].
    kp = movenet(tImage)
    kp_coords = np.squeeze(kp['output_0'].numpy())

    norm = {}

    for idx in range(len(kp_coords)):
        coord = get_cordinates_from(kp_coords[idx])
        norm[KEY_POINTS_NAMES[idx]] = [(coord[:2]), coord[2]]

    raw = {}
    for idx in range(len(kp_coords)):
        coord = kp_coords[idx]
        raw[KEY_POINTS_NAMES[idx]] = [(coord[:2]), coord[2]]

    return (norm, raw)

def draw_keypoints(image, kp):
    cv.circle(image, kp['nose'][0], 4, (0, 0, 255), -1)
    cv.circle(image, kp['left_eye'][0], 4, (255, 0, 0), -1)
    cv.circle(image, kp['right_eye'][0], 4, (255, 0, 0), -1)
    cv.circle(image, kp['left_shoulder'][0], 4, (0, 255, 0), -1)
    cv.circle(image, kp['right_shoulder'][0], 4, (0, 255, 0), -1)
    
    text = "Distance is ok!" if kp['left_shoulder'][1] > 0.5 and kp['right_shoulder'][1] > 0.5 else "Shoulders not visible"
    cv.putText(image, text, (20,40), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv.LINE_AA) 

    ## TODO rifattorizzare calcolo angolo rispetto asse Y + calcolare correttezza calcolo
    ## TODO calcolare angolo due punti rispetto asse X

    # HEAD VERTICAL ANGLE
    dividendo = (kp['right_eye'][0][1] - kp['left_eye'][0][1]) 
    divisore = math.sqrt((kp['right_eye'][0][0] - kp['left_eye'][0][0])**2 + (kp['right_eye'][0][1] - kp['left_eye'][0][1])**2)
    head_angle = 90 - round(math.degrees(math.acos(dividendo/divisore)))
    
    text = f"Head angle: {head_angle} L" if head_angle >= 0  else f"Head angle: {head_angle*-1} R"
    cv.putText(image, text, (20,80), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv.LINE_AA) 

    
    # SHOULDERS VERTICAL ANGLE
    dividendo = (kp['right_shoulder'][0][1] - kp['left_shoulder'][0][1]) 
    divisore = math.sqrt((kp['right_shoulder'][0][0] - kp['left_shoulder'][0][0])**2 + (kp['right_shoulder'][0][1] - kp['left_shoulder'][0][1])**2)
    shoulder_angle = 90 - round(math.degrees(math.acos(dividendo/divisore)))

    text = f"Shoulders angle: {shoulder_angle} L" if shoulder_angle >= 0  else f"Shoulders angle: {shoulder_angle*-1} R"
    cv.putText(image, text, (20,120), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv.LINE_AA) 

def draw_stats(image):
    text = f"{len(export_data)}"
    cv.putText(image, text, (20,600), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv.LINE_AA) 

# TODO Calcola inclinazione laterale testa
def compute_lateral_head_inclination(kp):
    # m = (y2-y1)/(x2-x1)
    eye_slope = (kp['right_eye'][0][1] - kp['left_eye'][0][1]) / (kp['right_eye'][0][0] - kp['left_eye'][0][0])
    eye_center = (((kp['right_eye'][0][0] + kp['left_eye'][0][0])/2), ((kp['right_eye'][0][1] + kp['left_eye'][0][1])/2))
    shoulder_center = (((kp['left_shoulder'][0][0] + kp['left_shoulder'][0][0])/2), ((kp['right_shoulder'][0][1] + kp['right_shoulder'][0][1])/2))
    pt1 = (int(eye_center[1] * eye_slope + eye_center[0]), 0)
    pt2 = (int((eye_center[1]-image_height) * eye_slope + eye_center[0]), image_height)
    myradians = math.atan2(eye_center[1]-shoulder_center[1], eye_center[0]-shoulder_center[0])
    mydegrees = math.degrees(myradians)

    #cv.line(image,  pt1, pt2, (0, 255, 0), 4)
    #cv.line(image,  kp['left_shoulder'][0],  kp['right_shoulder'][0], (0, 255, 0), 4)
    
    return 0

# TODO Calcola rotazione testa: con il modello 2D non posso calcolarlo
def compute_head_rotation(kp):
    
    return 0

# TODO Calcola inclinazione frontale testa: con il modello 2D non posso calcolarlo
def compute_frontal_head_inclination(kp):
    
    return 0

def add_to_keypoints(kp, pose, frame): 
    frame_uuid = uuid4()

    export_data.append({
        'pose' : pose,
        'frame': frame,
        'frame_uuid': frame_uuid,
        'date': datetime.now(),
        'nose' : kp['nose'],
        'left_eye' : kp['left_eye'], 
        'right_eye' : kp['right_eye'], 
        'left_ear' : kp['left_ear'], 
        'right_ear' : kp['right_ear'], 
        'left_shoulder' : kp['left_shoulder'], 
        'right_shoulder' : kp['right_shoulder'],
        'left_elbow' :  kp['left_elbow'], 
        'right_elbow' :  kp['right_elbow'], 
        'left_wrist' :  kp['left_wrist'], 
        'right_wrist' : kp['right_wrist'], 
        'left_hip' : kp['left_hip'], 
        'right_hip' : kp['right_hip'], 
        'left_knee' : kp['left_knee'], 
        'right_knee' : kp['right_knee'], 
        'left_ankle' : kp['left_ankle'], 
        'right_ankle' : kp['right_ankle']
    })

def export_keypoints(acquisition_info):
    print("[bold red]!!![/bold red] Export of recorded data [bold red]!!![/bold red]")

    exerciseId = acquisition_info["exercise_id"]
    exerciseFilename = acquisition_info["exercise"]["filename"]
    exercise_path =  os.path.join(DATASET_PATH, exerciseId)

    csvFile = os.path.join(exercise_path, f'{exerciseFilename}.csv')
    imagesFramePath = os.path.join(exercise_path, 'frames')

    #create folder if not exists
    make_folder(exercise_path)
    make_folder(imagesFramePath)

    has_headers = False
    if os.path.exists(csvFile):
        with open(csvFile, 'r') as f:
            first_line = f.readline()
            has_headers = len(first_line.strip()) > 0
        
    with Progress() as progress:
        task = progress.add_task("[green]Saving data...", total=len(export_data))

        with open(csvFile, 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)

            if not has_headers:
                # write the header
                writer.writerow([
                    'nose-x', 'nose-y', 'nose-p',
                    'left_eye-x', 'left_eye-y', 'left_eye-p',
                    'right_eye-x', 'right_eye-y', 'right_eye-p',
                    'left_ear-x', 'left_ear-y', 'left_ear-p',
                    'right_ear-x', 'right_ear-y', 'right_ear-p',
                    'left_shoulder-x', 'left_shoulder-y', 'left_shoulder-p',
                    'right_shoulder-x', 'right_shoulder-y', 'right_shoulder-p',
                    'left_elbow-x', 'left_elbow-y', 'left_elbow-p',
                    'right_elbow-x', 'right_elbow-y', 'right_elbow-p',
                    'left_wrist-x', 'left_wrist-y', 'left_wrist-p',
                    'right_wrist-x', 'right_wrist-y', 'right_wrist-p',
                    'left_hip-x', 'left_hip-y', 'left_hip-p',
                    'right_hip-x', 'right_hip-y', 'right_hip-p',
                    'left_knee-x', 'left_knee-y', 'left_knee-p',
                    'right_knee-x', 'right_knee-y', 'right_knee-p',
                    'left_ankle-x', 'left_ankle-y', 'left_ankle-p',
                    'right_ankle-x', 'right_ankle-y', 'right_ankle-p',

                    'camera-position',
                    'subject-position',
                    'class',
                    'camera',
                    'subject',
                    'model',
                    'date',
                    'frame-id',
                ])

            # write the data
            processed = 0
            for data in export_data:
                acquisition_timestamp = int(datetime.timestamp(data.get('date')))

                writer.writerow([
                    data.get('nose')[0][0], data.get('nose')[0][1], data.get('nose')[1],
                    data.get('left_eye')[0][0], data.get('left_eye')[0][1], data.get('left_eye')[1],
                    data.get('right_eye')[0][0], data.get('right_eye')[0][1], data.get('right_eye')[1],
                    data.get('left_ear')[0][0], data.get('left_ear')[0][1], data.get('left_ear')[1],
                    data.get('right_ear')[0][0], data.get('right_ear')[0][1], data.get('right_ear')[1],
                    data.get('left_shoulder')[0][0], data.get('left_shoulder')[0][1], data.get('left_shoulder')[1],
                    data.get('right_shoulder')[0][0], data.get('right_shoulder')[0][1], data.get('right_shoulder')[1],
                    data.get('left_elbow')[0][0], data.get('left_elbow')[0][1], data.get('left_elbow')[1],
                    data.get('right_elbow')[0][0], data.get('right_elbow')[0][1], data.get('right_elbow')[1],
                    data.get('left_wrist')[0][0], data.get('left_wrist')[0][1], data.get('left_wrist')[1],
                    data.get('right_wrist')[0][0], data.get('right_wrist')[0][1], data.get('right_wrist')[1],
                    data.get('left_hip')[0][0], data.get('left_hip')[0][1], data.get('left_hip')[1],
                    data.get('right_hip')[0][0], data.get('right_hip')[0][1], data.get('right_hip')[1],
                    data.get('left_knee')[0][0], data.get('left_knee')[0][1], data.get('left_knee')[1],
                    data.get('right_knee')[0][0], data.get('right_knee')[0][1], data.get('right_knee')[1],
                    data.get('left_ankle')[0][0], data.get('left_ankle')[0][1], data.get('left_ankle')[1],
                    data.get('right_ankle')[0][0], data.get('right_ankle')[0][1], data.get('right_ankle')[1],
                    
                    acquisition_info["camera_position"],
                    acquisition_info["subject_position"],
                    data.get('pose'),
                    acquisition_info["camera_id"],
                    acquisition_info["subject"],
                    acquisition_info["model"],
                    acquisition_timestamp,
                    str(data.get('frame_uuid')),
                ])

                #save frame
                frame_filename = os.path.join(imagesFramePath, f'frame_{acquisition_timestamp}_{str(data.get("frame_uuid"))}.png')
                cv.imwrite(frame_filename, data.get('frame'))

                # Update
                processed = processed + 1
                progress.update(task, completed=processed)

    print(f"Exported {len(export_data)} records")
    print(f"Total records: {count_rows(csvFile) - 1}")