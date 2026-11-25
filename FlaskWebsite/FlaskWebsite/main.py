from flask import Flask, render_template, request  
import pandas as pd 
import joblib 

app = Flask(__name__)

MODEL_PATH = "car_price_model.pkl"
SCALER_PATH = "car_price_scaler.pkl"

# load the model 
model = joblib.load(MODEL_PATH)
# load the scaler 
scaler = joblib.load(SCALER_PATH)

@app.route("/")
def index(): 
    return render_template("index.html")

@app.route("/predict-price", methods=["POST"])
def predict_price(): 
    year = int(request.form["year"]) 
    mileage = int(request.form["mileage"])
    X = pd.DataFrame([[year, mileage]], columns=["Year", "Mileage"])
    # scaled features 
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return render_template("index.html", result=prediction)

