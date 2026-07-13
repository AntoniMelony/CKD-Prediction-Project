import pickle
import pandas as pd
import streamlit as st

# Load saved model files
ckd_model = pickle.load(open("ckd_model.sav", "rb"))
label_encoder = pickle.load(open("ckd_label_encoder.sav", "rb"))
feature_columns = pickle.load(open("ckd_feature_columns.sav", "rb"))

st.set_page_config(page_title="Chronic Kidney Disease Prediction", layout="wide")

st.title("Chronic Kidney Disease Prediction System")
st.subheader("Enter patient details below")

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    age = st.text_input("Age")
    bp = st.text_input("Blood Pressure")
    sg = st.selectbox("Specific Gravity", ["", "1.005", "1.010", "1.015", "1.020", "1.025"])
    al = st.selectbox("Albumin", ["", "0", "1", "2", "3", "4", "5"])
    su = st.selectbox("Sugar", ["", "0", "1", "2", "3", "4", "5"])
    rbc = st.selectbox("Red Blood Cells", ["", "normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["", "normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["", "present", "notpresent"])

with col2:
    ba = st.selectbox("Bacteria", ["", "present", "notpresent"])
    bgr = st.text_input("Blood Glucose Random")
    bu = st.text_input("Blood Urea")
    sc = st.text_input("Serum Creatinine")
    sod = st.text_input("Sodium")
    pot = st.text_input("Potassium")
    hemo = st.text_input("Hemoglobin")
    pcv = st.text_input("Packed Cell Volume")

with col3:
    wc = st.text_input("White Blood Cell Count")
    rc = st.text_input("Red Blood Cell Count")
    htn = st.selectbox("Hypertension", ["", "yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", ["", "yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", ["", "yes", "no"])
    appet = st.selectbox("Appetite", ["", "good", "poor"])
    pe = st.selectbox("Pedal Edema", ["", "yes", "no"])
    ane = st.selectbox("Anemia", ["", "yes", "no"])


def to_float(value):
    if value == "" or value is None:
        return None
    return float(value)


diagnosis = ""

if st.button("CKD Test Result"):
    try:
        input_dict = {
            "age": to_float(age),
            "bp": to_float(bp),
            "sg": to_float(sg) if sg != "" else None,
            "al": to_float(al) if al != "" else None,
            "su": to_float(su) if su != "" else None,
            "rbc": rbc if rbc != "" else None,
            "pc": pc if pc != "" else None,
            "pcc": pcc if pcc != "" else None,
            "ba": ba if ba != "" else None,
            "bgr": to_float(bgr),
            "bu": to_float(bu),
            "sc": to_float(sc),
            "sod": to_float(sod),
            "pot": to_float(pot),
            "hemo": to_float(hemo),
            "pcv": to_float(pcv),
            "wc": to_float(wc),
            "rc": to_float(rc),
            "htn": htn if htn != "" else None,
            "dm": dm if dm != "" else None,
            "cad": cad if cad != "" else None,
            "appet": appet if appet != "" else None,
            "pe": pe if pe != "" else None,
            "ane": ane if ane != "" else None
        }

        input_df = pd.DataFrame([input_dict])

        # Ensure exact same column order
        input_df = input_df.reindex(columns=feature_columns)

        prediction = ckd_model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        if predicted_label == "ckd":
            diagnosis = "The person is likely to have Chronic Kidney Disease."
        else:
            diagnosis = "The person is not likely to have Chronic Kidney Disease."

    except ValueError:
        diagnosis = "Please enter valid numeric values in the numeric fields."

st.success(diagnosis)