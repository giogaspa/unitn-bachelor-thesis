import os
import csv
from rich import print
from rich.progress import Progress


def make_folder(path):
    # checking if the directory demo_folder  
    # exist or not. 
    if not os.path.exists(path): 
        # if the demo_folder directory is not present  
        # then create it. 
        os.makedirs(path)

def count_rows(file_path):
    # open file in read mode
    with open(file_path, 'r', encoding='UTF8') as fp:
        for count, line in enumerate(fp):
            pass
    return count + 1

def write_csv(file_path, data):
    with Progress() as progress:
        task = progress.add_task("[green]Saving data...", total=len(data))

        with open(file_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            processed = 0
            for r in data:
                writer.writerow(r)

                # Update
                processed = processed + 1
                progress.update(task, completed=processed)

    print(f"Saved {len(data)} records")

def list_files_in_folder(path):
    res = []

    for file_path in os.listdir(path):
    # check if current file_path is a file
        if os.path.isfile(os.path.join(path, file_path)) and file_path.endswith('keras'):
            # add filename to list
            res.append(file_path)
    
    return res