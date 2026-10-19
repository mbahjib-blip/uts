"""Training Decision Tree - klasifikasi diabetes. Jalankan: python train.py"""
import json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

FITUR = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
ZERO_NAN = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

df = pd.read_csv("data/diabetes.csv", sep=";")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df.columns = [c.replace(" ", "") for c in df.columns]
info = {"baris_awal": len(df), "duplikat": int(df.duplicated().sum()),
        "nol_per_kolom": {c: int((df[c] == 0).sum()) for c in ZERO_NAN}}

def jalankan(data, tag):
    d = data.copy(); d[ZERO_NAN] = d[ZERO_NAN].replace(0, np.nan)
    Xtr, Xte, ytr, yte = train_test_split(d[FITUR], d["Outcome"], test_size=0.2, stratify=d["Outcome"], random_state=42)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("dt", DecisionTreeClassifier(random_state=42))])
    gs = GridSearchCV(pipe, {"dt__max_depth": [3,4,5,6,8,None], "dt__min_samples_leaf": [1,3,5,10], "dt__criterion": ["gini","entropy"]},
                      cv=5, scoring="accuracy").fit(Xtr, ytr)
    p = gs.predict(Xte)
    m = {"accuracy": accuracy_score(yte,p), "precision": precision_score(yte,p), "recall": recall_score(yte,p),
         "cm": confusion_matrix(yte,p).tolist(), "n_train": len(Xtr), "n_test": len(Xte), "params": gs.best_params_}
    return gs.best_estimator_, m, yte, p

# Pembanding: TANPA hapus duplikat (data bocor antara train & test -> akurasi menggembung)
_, m_dup, _, _ = jalankan(df, "dup")
# Utama: duplikat dihapus
bersih = df.drop_duplicates().reset_index(drop=True)
model, m, yte, p = jalankan(bersih, "bersih")
info.update({"baris_bersih": len(bersih), "kelas": {int(k): int(v) for k, v in bersih["Outcome"].value_counts().items()}})

joblib.dump(model, "model.joblib")
medians = bersih[FITUR].replace({c: {0: np.nan} for c in ZERO_NAN}).median().round(2).to_dict()
json.dump({"info": info, "metrics": m, "metrics_dengan_duplikat": m_dup, "median": medians},
          open("metrics.json", "w"), indent=2, default=float)

ConfusionMatrixDisplay(confusion_matrix(yte, p), display_labels=["Tidak","Diabetes"]).plot(cmap="Blues")
plt.title("Confusion Matrix (data uji)"); plt.savefig("static/cm.png", dpi=110, bbox_inches="tight"); plt.close()
imp = pd.Series(model.named_steps["dt"].feature_importances_, index=FITUR).sort_values()
imp.plot.barh(color="#2563eb"); plt.title("Feature Importance"); plt.savefig("static/fi.png", dpi=110, bbox_inches="tight"); plt.close()
plt.figure(figsize=(16,7)); plot_tree(model.named_steps["dt"], feature_names=FITUR, class_names=["Tidak","Diabetes"], filled=True, max_depth=3, fontsize=8)
plt.savefig("static/tree.png", dpi=100, bbox_inches="tight"); plt.close()
print(json.dumps({"info": info, "metrics": m, "dengan_duplikat": m_dup}, indent=1, default=float))
