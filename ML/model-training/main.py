import sys
import os
import pathlib
import calendar
import time
import typer
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich import print

# Load common module path
_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(_dir, '..'))
sys.path.append(os.path.join(_dir, 'models'))

from common.params import exercises, DATASET_PATH
import ex1

def main():
    ## Prompt data
    exerciseId = inquirer.select(
        message="Choose the exercise:",
        choices= [Choice(value=id, name=exercises[id]["name"]) for id in exercises]
    ).execute()

    layers = int(inquirer.text(
        message="Enter layers number:",
    ).execute())

    layers_configuration = []
    for idx in range(layers):
        units = int(inquirer.text(
            message=f"Enter layer-{idx + 1} units number:",
        ).execute())
        layers_configuration.append(units)

    units_text = "_".join(map(str, layers_configuration))
    model_name = f"L{layers}_U{units_text}"

    model_configuration = {
        "name": model_name,
        "layers": layers,
        "units": units_text,
        "configuration": layers_configuration
    }

    ## Train
    model = None
    match exerciseId:
        case 'ex1':
            model = ex1.train(model_configuration)
        case _:
            print(f"[red]Missing exercise '{exerciseId}' training model.[/red]")


    if model is not None:
        # SAVE MODEL
        save_model = inquirer.confirm("Save model?", default=True).execute()

        if save_model:
            current_GMT = time.gmtime()
            time_stamp = str(calendar.timegm(current_GMT))
            name = inquirer.text("Model name:", default=f"{time_stamp}_{model_name}").execute()

            model_filepath = os.path.join(DATASET_PATH, exerciseId, 'models', f'{exercises[exerciseId]["filename"]}_{name}.keras')
            model.save(model_filepath)

            model_filepath = os.path.join(DATASET_PATH, exerciseId, 'models', f'{exercises[exerciseId]["filename"]}_{name}.h5')
            model.save(model_filepath, save_format='h5')

if __name__ == "__main__":
    typer.run(main)