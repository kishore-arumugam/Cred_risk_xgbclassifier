import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OneHotEncoder as OHE
import joblib as jl
import yaml
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
yml = yaml.safe_load(open(BASE_DIR / "config.yaml"))

def proc_data(path):
    df = pd.read_csv(path)
    df['Income_to_Loan_Ratio'] = df['Annual_Income'] / df['Loan_Amount']
    df['Monthly_Income'] = df['Annual_Income'] / 12
    df['Est_Monthly_Debt'] = (df['Debt_to_Income'] / 100) * df['Monthly_Income']
    df['Disposable_Income'] = df['Monthly_Income'] - df['Est_Monthly_Debt']
    enc = OHE(sparse_output=False, handle_unknown='ignore')
    cats = ['Home_Ownership']
    ec = enc.fit_transform(df[cats])
    fns = enc.get_feature_names_out(cats)
    edf = pd.DataFrame(ec, columns=fns)
    df = pd.concat([df, edf], axis=1)
    df.drop(columns=cats, inplace=True)
    jl.dump(enc, BASE_DIR / yml["enc_path"])
    return df

def splt(df, tgt='Default', sz=0.2):
    X = df.drop(columns=[tgt])
    y = df[tgt]
    xt, xv, yt, yv = tts(X, y, test_size=sz, random_state=42, stratify=y)
    return xt, xv, yt, yv

if __name__ == "__main__":
    p = BASE_DIR / yml["sub_path"]["data"]
    df = proc_data(p)
    print(f"Done.\n{df.head()}")
    xt, xv, yt, yv = splt(df)
    print(f"Tr: {xt.shape}, Te: {xv.shape}")
    xt.to_csv(BASE_DIR / yml["sub_path"]["X_train"], index=False)
    xv.to_csv(BASE_DIR / yml["sub_path"]["X_test"], index=False)
    yt.to_csv(BASE_DIR / yml["sub_path"]["y_train"], index=False)
    yv.to_csv(BASE_DIR / yml["sub_path"]["y_test"], index=False)