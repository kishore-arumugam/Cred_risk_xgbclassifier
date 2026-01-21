import streamlit as st
import pandas as pd
import numpy as np
import joblib as jl
import shap
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
yml = yaml.safe_load(open(BASE_DIR / "config.yaml"))

@st.cache_resource
def ld():
    return jl.load(BASE_DIR / yml["mdl_path"]), jl.load(BASE_DIR / yml["enc_path"])

mdl, enc = ld()
st.set_page_config(page_title="BFSI Credit Risk Assessment", layout="wide")
st.title("🏦 BFSI Credit Risk Assessment System")
st.markdown("Predicts **Loan Default** using XGBoost & SHAP.")
st.sidebar.header("Applicant Information")

def get_in():
    inc = st.sidebar.number_input("Annual Income ($)", 10000, 1000000, 60000)
    emp = st.sidebar.slider("Employment Length (Years)", 0, 40, 5)
    dti = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 20.0)
    csc = st.sidebar.slider("Credit Score (FICO)", 300, 850, 700)
    amt = st.sidebar.number_input("Loan Amount ($)", 1000, 50000, 15000)
    trm = st.sidebar.selectbox("Loan Term (Months)", (36, 60))
    own = st.sidebar.selectbox("Home Ownership", ("RENT", "MORTGAGE", "OWN"))
    return pd.DataFrame({'Annual_Income': inc, 'Employment_Length': emp, 'Debt_to_Income': dti, 'Credit_Score': csc, 'Loan_Amount': amt, 'Loan_Term': trm, 'Home_Ownership': own}, index=[0])

idf = get_in()
idf['Income_to_Loan_Ratio'] = idf['Annual_Income'] / idf['Loan_Amount']
idf['Monthly_Income'] = idf['Annual_Income'] / 12
idf['Est_Monthly_Debt'] = (idf['Debt_to_Income'] / 100) * idf['Monthly_Income']
idf['Disposable_Income'] = idf['Monthly_Income'] - idf['Est_Monthly_Debt']
cats = ['Home_Ownership']
ec = enc.transform(idf[cats])
fns = enc.get_feature_names_out(cats)
edf = pd.DataFrame(ec, columns=fns)
xt = pd.read_csv(BASE_DIR / yml["sub_path"]["X_train"], nrows=1)
inf = pd.concat([idf, edf], axis=1).drop(columns=cats)[xt.columns]

if st.button("Assess Risk"):
    pred = mdl.predict(inf)
    prob = mdl.predict_proba(inf)[:, 1][0]
    st.subheader("Assessment Result")
    c1, c2 = st.columns(2)
    with c1:
        if pred[0] == 1: st.error(f"⚠️ High Risk: Default Predicted")
        else: st.success(f"✅ Low Risk: Approved")
    with c2: st.metric("Default Probability", f"{prob:.2%}")
    st.progress(float(prob))
    st.subheader("Why this decision?")
    st.write("Feature contributions (SHAP):")
    exp = shap.TreeExplainer(mdl)
    sv = exp(inf)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig)
    st.info("**Red**: Higher Risk | **Blue**: Lower Risk")