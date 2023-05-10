#!/usr/bin/env python3

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys

DATASET = "Texas_Inpatient_Discharge"
SPLIT_TRAINING = True
DEBUG = False
SEED = 42

COLAB = 'google.colab' in sys.modules
if COLAB:
    ROOT = f"/content/gdrive/MyDrive/datasets/{DATASET.replace(' ','_')}/"
else:
    ROOT = "./"

df = pd.read_pickle(f"{ROOT}/data/df_train_preprocess_00_of_10.pkl")
df.dropna(inplace=True)
#df = df.loc[df.sex!='.'].copy()

model = joblib.load("model.sav")


# Get the columns to be used as sliders
slider_columns = [
   "TYPE_OF_ADMISSION",
    "SOURCE_OF_ADMISSION",
    "PAT_STATE",
    "PUBLIC_HEALTH_REGION",
    "ADMITTING_DIAGNOSIS",
    "PRINC_DIAG_CODE",
    "POA_PRINC_DIAG_CODE",
    "ADMIT_WEEKDAY"
]

# Create the Streamlit app
st.write("# Texas Inpatient Discharge")
st.write("## Enter case details:")

# Create sliders for each column
case = {}
for column in slider_columns:
    unique_values = df[column].unique()
    selected_value = st.selectbox(column, unique_values)
    case[column] = selected_value

st.write("## Prediction")
st.write("Case details:")

case_array = np.array(list(case.values())).reshape(1, -1)
case_df = pd.DataFrame(case_array, columns=slider_columns)



st.dataframe(case)

st.write("## Prediction")

y_pred = model.predict(case_df)
y_pred_proba = model.predict_proba(case_df)

st.write(f"""
 * Predicted stay is {y_pred[0]} with probability of {y_pred_proba[0].max()}.
""")