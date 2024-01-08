import sys
import os
import pathlib
import numpy as np
import typer
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import EmptyInputValidator
from rich import print

# Load common module path
_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(_dir, '..'))
sys.path.append(os.path.join(_dir, 'data-processors'))

from common.params import exercises, DATASET_PATH
from common.file import write_csv
import ex1

def main():
    ## Scegli l'esercizio da processare
    ## Prompt data
    exerciseId = inquirer.select(
        message="Choose the exercise to process:",
        choices= [Choice(value=id, name=exercises[id]["name"]) for id in exercises]
    ).execute()

    ## Load data
    dataset_filepath = os.path.join(DATASET_PATH, exerciseId, f'{exercises[exerciseId]["filename"]}.csv')
    raw_data = np.genfromtxt(dataset_filepath, delimiter=",", dtype=None, encoding='UTF8')
    raw_data = raw_data[1:,:54] # Remove header row and useless column

    processed_data = None

    ## Process data 
    match exerciseId:
        case 'ex1':
            processed_data = ex1.process(raw_data)
        case _:
            print(f"[red]Missing exercise '{exerciseId}' data preprocessor.[/red]")


    ## Save processed data
    if processed_data is not None:
        file_path = os.path.join(DATASET_PATH, exerciseId, f'{exercises[exerciseId]["filename"]}_processed.csv')
        write_csv(file_path, processed_data.tolist())

if __name__ == "__main__":
    typer.run(main)