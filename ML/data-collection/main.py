import cv2 as cv
import tensorflow as tf
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import EmptyInputValidator
import typer
from rich import print

from params import *
from utils import *

def main():

    ## Setup capturing session
    ## Prompt data
    exerciseId = inquirer.select(
        message="Choose exercise:",
        choices= [Choice(value=id, name=exercises[id]["name"]) for id in exercises]
    ).execute()

    exercise = exercises[exerciseId]
    acquisition_info["exercise_id"] = exerciseId
    acquisition_info["exercise"] = exercise

    acquisition_info["subject"] = inquirer.select(
        message="Choose subject:",
        choices= subject
    ).execute()

    acquisition_info["subject_position"] = inquirer.select(
        message="Choose subject position:",
        choices= subject_position
    ).execute()

    acquisition_info["camera_id"] = inquirer.select(
        message="Choose camera:",
        choices= [Choice(value=camera["id"], name=camera["name"]) for camera in camera_list]
    ).execute()

    acquisition_info["camera_position"] = inquirer.select(
        message="Choose camera position:",
        choices= camera_position
    ).execute()

    cameraIndex = inquirer.number(
        message="Enter camera index:",
        min_allowed=0,
        max_allowed=10,
        validate=EmptyInputValidator(),
    ).execute()

    ## Capture video streaming from webcam
    cap = cv.VideoCapture(int(cameraIndex))

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

        # Resize frame
        #flippedImage = cv.flip(frame, 1)
        resizedImage = tf.cast(tf.image.resize_with_pad(frame, image_height, image_width), dtype=tf.uint8).numpy()

        # Draw keypoints
        draw_keypoints(resizedImage, kp_norm)
        draw_stats(resizedImage)

        # Display the resulting frame
        cv.imshow('Webcam', resizedImage)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        command = next((item for item in exercise['captureCommands'] if item["key"] == key), False)
        if(command):
            add_to_keypoints(kp_raw, command["className"], frame)


    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(100)

    export_keypoints(acquisition_info)

if __name__ == "__main__":
    typer.run(main)