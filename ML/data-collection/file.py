import os


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