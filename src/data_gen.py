import pandas as pd
import numpy as np
import random as rd
import yaml
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
yml = yaml.safe_load(open(BASE_DIR / "config.yaml"))    
def gen_data(n=10000, s=42):
    np.random.seed(s)
    rd.seed(s)
    inc = np.round(np.random.lognormal(11, 0.5, n), 2)
    emp = np.random.randint(0, 41, n)
    dti = np.round(np.clip(np.random.normal(20, 10, n), 0, 100), 2)
    base = np.random.normal(650, 50, n)
    ifac = (np.log(inc) - 10) * 20
    csc = np.round(np.clip(base + ifac, 300, 850), 0)
    amt = (np.random.randint(1000, 50000, n) // 100) * 100
    term = np.random.choice([36, 60], n, p=[0.7, 0.3])
    own = np.random.choice(['RENT', 'MORTGAGE', 'OWN'], n, p=[0.4, 0.5, 0.1])
    n_dti = dti / 100.0
    n_csc = (850 - csc) / 550.0
    n_inc = 1.0 - (np.log(inc) / 15.0)
    risk = 0.4 * n_dti + 0.4 * n_csc + 0.2 * n_inc + np.random.normal(0, 0.1, n)
    prob = 1 / (1 + np.exp(-((risk - 0.5) * 10))) 
    tgt = (np.random.rand(n) < prob).astype(int)
    return pd.DataFrame({'Annual_Income': inc, 'Employment_Length': emp, 'Debt_to_Income': dti, 'Credit_Score': csc, 'Loan_Amount': amt, 'Loan_Term': term, 'Home_Ownership': own, 'Default': tgt})

if __name__ == "__main__":
    print("Generating...")
    df = gen_data()
    print(f"Shape: {df.shape}\nDef Rate: {df['Default'].mean():.2%}")
    path = BASE_DIR / yml["sub_path"]["data"]
    df.to_csv(path, index=False)
    print(f"Saved to {path}")