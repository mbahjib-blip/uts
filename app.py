import json, os, joblib, numpy as np, pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
model = joblib.load("model.joblib")
M = json.load(open("metrics.json"))
FITUR = [("Pregnancies","Jumlah kehamilan"),("Glucose","Glukosa (mg/dL)"),("BloodPressure","Tekanan darah diastolik (mm Hg)"),
         ("SkinThickness","Tebal lipatan kulit (mm)"),("Insulin","Insulin (mu U/ml)"),("BMI","BMI"),
         ("DiabetesPedigreeFunction","Diabetes Pedigree Function"),("Age","Umur (tahun)")]
ZERO_NAN = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

@app.route("/")
def index():
    sample = pd.read_csv("data/diabetes.csv", sep=";").iloc[:, :9].head(8).to_html(index=False, classes="tbl")
    return render_template("index.html", M=M, sample=sample)

@app.route("/evaluasi")
def evaluasi():
    return render_template("evaluasi.html", M=M)

@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    hasil, nilai = None, {}
    if request.method == "POST":
        try:
            nilai = {k: float(request.form[k]) for k, _ in FITUR}
            x = pd.DataFrame([nilai])
            x[ZERO_NAN] = x[ZERO_NAN].replace(0, np.nan)   # 0 dianggap tidak diketahui -> diisi median
            p = int(model.predict(x)[0]); pr = float(model.predict_proba(x)[0][p])
            hasil = {"label": "Terindikasi DIABETES" if p else "TIDAK terindikasi diabetes", "p": p, "prob": round(pr*100, 1)}
        except ValueError:
            hasil = {"error": "Semua kolom harus diisi angka."}
    return render_template("prediksi.html", FITUR=FITUR, hasil=hasil, nilai=nilai)

@app.route("/perbandingan")
def perbandingan():
    return render_template("perbandingan.html", M=M)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
