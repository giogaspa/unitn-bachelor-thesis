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

CLASS_TRANSFORMER = [
    "center",
    "wrong-center-forward",
    "wrong-center-backward",
    "right",
    "wrong-right-forward",
    "wrong-right-backward",
    "left",
    "wrong-left-forward",
    "wrong-left-backward",
]

def main():
    ## Prompt data
    exerciseId = inquirer.select(
        message="Choose the exercise to load:",
        choices= [Choice(value=id, name=exercises[id]["name"]) for id in exercises]
    ).execute()

    ## Load model
    model_filepath = os.path.join(DATASET_PATH, exerciseId, f'{exercises[exerciseId]["filename"]}.keras')
    model = tf.keras.models.load_model(model_filepath)

    ## Capture video streaming from webcam
    cap = cv.VideoCapture(int(0))

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("[bold red]Can't receive frame (stream end?). Exiting ...[/bold red]")
            break
        
        frame = cv.flip(frame, 1)

        # Our operations on the frame come here
        kp_norm, kp_raw = compute_key_points(frame)

        data = np.array([
            kp_raw['nose'][0][0],kp_raw['nose'][0][1],
            kp_raw['left_eye'][0][0],kp_raw['left_eye'][0][1],
            kp_raw['right_eye'][0][0],kp_raw['right_eye'][0][1],
            kp_raw['left_ear'][0][0],kp_raw['left_ear'][0][1],
            kp_raw['right_ear'][0][0],kp_raw['right_ear'][0][1],
            kp_raw['left_shoulder'][0][0],kp_raw['left_shoulder'][0][1],
            kp_raw['right_shoulder'][0][0],kp_raw['right_shoulder'][0][1],
        ])

        data = tf.constant(data.astype(np.float32).reshape([1, 14]))
        predicted = model.predict(data, verbose=0)
        predicted = tf.nn.softmax(predicted).numpy()
        predicted_class = CLASS_TRANSFORMER[np.argmax(predicted)]

        # Resize frame
        #flippedImage = cv.flip(frame, 1)
        resizedImage = tf.cast(tf.image.resize_with_pad(frame, image_height, image_width), dtype=tf.uint8).numpy()

        cv.putText(resizedImage, predicted_class, (20,40), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv.LINE_AA) 

        # Display the resulting frame
        cv.imshow('Webcam', resizedImage)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break


    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(100)

if __name__ == "__main__":
    typer.run(main)