# Collection of functions useful for analysing TeXas InPateint Dataset 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
sns.set_style("darkgrid")

from IPython.display import display, Markdown
pd.set_option('display.max_columns', None)  

import glob, os, sys

DATASET = "Texas_Inpatient_Discharge"
SPLIT_TRAINING = True
DEBUG = False
SEED = 42

COLAB = 'google.colab' in sys.modules
if COLAB:
    ROOT = f"/content/gdrive/MyDrive/datasets/{DATASET.replace(' ','_')}/"
else:
    ROOT = "./"


from zipfile import ZipFile
import os

def make_assignment(files=None, archive="my_assignment.zip"):
    """Create assignment archive for uploading to Moodle. Optional parameter `files` can be a list of additional files to include."""
    default_files = {"01-Import.ipynb", "02-EDA.ipynb", "03-Model.ipynb", "my_lib.py", "df_grading_pred.csv", "dashboard.py"}

    if files is None:
        files = default_files
    else:
        if type(files)==str: files = [files]
        if type(files) in [list,tuple]: files = set(files)
        files = files.union(default_files)

    print(f"Creating archive: {archive}")
    with ZipFile(f"{ROOT}/{archive}","w") as zip:
        for f in files:
            if os.path.isfile(f"{ROOT}/{f}"):
                print(f"\t{f} - OK")
                zip.write(f) 
            else:
                print(f"\t{f} - Skipped (missing file)")


################################################################
# Put your code below this block
################################################################





