import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("svc.pkl", "rb"))
le = pickle.load(open("le.pkl", "rb"))
le.classes_ = np.load("classes.npy", allow_pickle=True)
test_data = pd.read_csv("data.csv")
test_data = test_data.drop("Unnamed: 0", axis=1)

st.title("Disease Prediction")
symptoms = st.text_area("Symptoms")

if st.button("Predict"):
    symptoms = symptoms.split(",")
    test_index = [test_data.columns.get_loc(col) for col in symptoms]
    for index in test_index:
        test_data.loc[0, test_data.columns[index]] = 1

    test_data.fillna(0, inplace=True)
    test = np.array(test_data)
    res = model.predict(test)
    print(res)
    result = le.classes_[res[0]]
    st.success("Predicted: " + str(result))
