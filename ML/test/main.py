import sys
import os
import pathlib
import cv2 as cv
import numpy as np
import tensorflow as tf
from InquirerPy import inquirer
from rich import print
from InquirerPy.base.control import Choice
import typer

# Load common module path
_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(_dir, '..'))
sys.path.append(os.path.join(_dir, 'models'))

from common.params import exercises, DATASET_PATH, image_height, image_width
from common.utils import compute_key_points
from common.file import list_files_in_folder

import ex1

def main():
    ## Prompt data
    exerciseId = inquirer.select(
        message="Choose the exercise to load:",
        choices= [Choice(value=id, name=exercises[id]["name"]) for id in exercises]
    ).execute()

    exercise_models_path = os.path.join(DATASET_PATH, exerciseId, 'models')
    models = list_files_in_folder(exercise_models_path)

    model_idx = inquirer.select(
        message="Choose the model to load:",
        choices= [Choice(value=idx, name=model) for idx, model in enumerate(models)]
    ).execute()

    model_filename = models[model_idx]
    model_filepath = os.path.join(exercise_models_path, model_filename)

    match exerciseId:
        case 'ex1':
            ex1.load_model(model_filepath)


    ## Capture video streaming from webcam
    cap = cv.VideoCapture(int(0))

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("[bold red]Can't receive frame (stream end?). Exiting ...[/bold red]")
            break

        # Our operations on the frame come here
        kp_norm, kp_raw = compute_key_points(frame)

        prediction = None
        match exerciseId:
            case 'ex1':
                prediction = ex1.predict(kp_raw)

        # Resize frame
        resizedImage = tf.cast(tf.image.resize_with_pad(frame, image_height, image_width), dtype=tf.uint8).numpy()

        cv.putText(resizedImage, prediction, (20,40), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv.LINE_AA) 

        # Display the resulting frame
        cv.imshow('Webcam', resizedImage)
        
        key = cv.waitKey(1) & 0xFF
        if key == 27: #esc
            break


    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(100)

if __name__ == "__main__":
    typer.run(main)