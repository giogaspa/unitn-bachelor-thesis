import sys
import os
import pathlib
import numpy as np
import typer
import tensorflow as tf
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import EmptyInputValidator
from rich import print

# Load common module path
_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(_dir, '..'))
sys.path.append(os.path.join(_dir, 'models'))

from common.params import exercises, DATASET_PATH
import ex1

CLASS_TRANSFORMER = {
    "center": 0,
    "wrong-center-forward": 1,
    "wrong-center-backward": 2,
    "right": 3,
    "wrong-right-forward": 4,
    "wrong-right-backward": 5,
    "left": 6,
    "wrong-left-forward": 7,
    "wrong-left-backward": 8,
}

def main():
    ## Prompt data
    # TODO: chiedere all'utente nome del test, salvare parametri del modello e risultati ottenuti

    exerciseId = inquirer.select(
        message="Choose the exercise:",
        choices= [Choice(value=id, name=exercises[id]["name"]) for id in exercises]
    ).execute()

    ## Load data
    dataset_filepath = os.path.join(DATASET_PATH, exerciseId, f'{exercises[exerciseId]["filename"]}_processed.csv')
    raw_data = np.genfromtxt(dataset_filepath, delimiter=",", dtype=None, encoding='UTF8')
    raw_data = raw_data[1:,:]

    ## Split into train and test data
    train_data, test_data = np.split(raw_data,[int(0.8 * len(raw_data))])
    # data_train, data_test = tf.keras.utils.split_dataset(raw_data, left_size=0.8, shuffle=True)

    ## Train
    model = None
    match exerciseId:
        case 'ex1':
            model = ex1.train(train_data)
            # test accuracy
        case _:
            print(f"[red]Missing exercise '{exerciseId}' training model.[/red]")


    ## Export model
    if model is not None:
        test_values = test_data[:,:14]
        test_values = test_values.astype(np.float32)
        test_values = tf.constant(test_values)

        new_labels = []
        for l in test_data[:,14]:
            new_labels.append(CLASS_TRANSFORMER[l])

        test_labels = np.array(new_labels).reshape([len(test_data), 1])
        test_labels = tf.constant(test_labels)

        print("[red]Evaluate model[/red]")
        model.evaluate(test_values, test_labels)

        print("[red]Predict 1[/red]")
        print(test_data[0][14])
        prepared_data = tf.constant(test_data[0,:14].astype(np.float32).reshape([1, 14]))
        predicted = model.predict(prepared_data)
        predicted = tf.nn.softmax(predicted).numpy()
        print(np.argmax(predicted))
        
        print("[red]Predict 2[/red]")
        print(test_data[2][14])
        prepared_data = tf.constant(test_data[2,:14].astype(np.float32).reshape([1, 14]))
        predicted = model.predict(prepared_data)
        predicted = tf.nn.softmax(predicted).numpy()
        print(np.argmax(predicted))

        print("[red]Predict 3[/red]")
        print(test_data[10][14])
        prepared_data = tf.constant(test_data[10,:14].astype(np.float32).reshape([1, 14]))
        predicted = model.predict(prepared_data)
        predicted = tf.nn.softmax(predicted).numpy()
        print(np.argmax(predicted))

        print("[red]Predict 4[/red]")
        print(test_data[20][14])
        prepared_data = tf.constant(test_data[20,:14].astype(np.float32).reshape([1, 14]))
        predicted = model.predict(prepared_data)
        predicted = tf.nn.softmax(predicted).numpy()
        print(np.argmax(predicted))


        print("[red]Predict 5[/red]")
        print(test_data[30][14])
        prepared_data = tf.constant(test_data[30,:14].astype(np.float32).reshape([1, 14]))
        predicted = model.predict(prepared_data)
        predicted = tf.nn.softmax(predicted).numpy()
        print(np.argmax(predicted))

        print("[red]Predict 6[/red]")
        print(test_data[38][14])
        prepared_data = tf.constant(test_data[38,:14].astype(np.float32).reshape([1, 14]))
        predicted = model.predict(prepared_data)
        predicted = tf.nn.softmax(predicted).numpy()
        print(np.argmax(predicted))

        # SAVE MODEL
        save_model = inquirer.confirm("Save model?", default=True).execute()

        if save_model:
            model_filepath = os.path.join(DATASET_PATH, exerciseId, f'{exercises[exerciseId]["filename"]}.keras')
            model.save(model_filepath)

            model_filepath = os.path.join(DATASET_PATH, exerciseId, f'{exercises[exerciseId]["filename"]}.h5')
            model.save(model_filepath, save_format='h5')

if __name__ == "__main__":
    typer.run(main)