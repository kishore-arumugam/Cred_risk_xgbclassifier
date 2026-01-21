import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report as cr, roc_auc_score as ras
import shap
import joblib as jl
import yaml
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
yml = yaml.safe_load(open(BASE_DIR / "config.yaml"))
def trn():
    
    xt = pd.read_csv(BASE_DIR / yml["sub_path"]["X_train"])
    xv = pd.read_csv(BASE_DIR / yml["sub_path"]["X_test"])
    yt = pd.read_csv(BASE_DIR / yml["sub_path"]["y_train"]).values.ravel()
    yv = pd.read_csv(BASE_DIR / yml["sub_path"]["y_test"]).values.ravel()
    pc = sum(yt)
    nc = len(yt) - pc
    spw = nc / pc
    print(f"Imbal: P={pc}, N={nc}, W={spw:.2f}")
    mdl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=200, learning_rate=0.05, max_depth=5, scale_pos_weight=spw, use_label_encoder=False, eval_metric='logloss', random_state=42)
    print("Training...")
    mdl.fit(xt, yt)
    yp = mdl.predict(xv)
    pb = mdl.predict_proba(xv)[:, 1]
    print(f"\nRpt:\n{cr(yv, yp)}")
    sc = ras(yv, pb)
    print(f"AUC: {sc:.4f}")
    jl.dump(mdl, BASE_DIR / yml["mdl_path"])
    print("Saved mdl.")
    print("SHAP...")
    exp = shap.Explainer(mdl)
    sv = exp(xv)
    jl.dump(exp, BASE_DIR / yml["shap_path"])

if __name__ == "__main__":
    trn()