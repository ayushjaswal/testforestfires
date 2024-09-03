import pickle
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#importing ridge and standard scaler
ridge = pickle.load(open("models/ridge.pkl", 'rb'))
scaler = pickle.load(open("models/scaler.pkl", 'rb'))
application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict_data", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge.predict(data)

        return render_template('predict.html', result=result[0])
    else:
        return render_template("predict.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
