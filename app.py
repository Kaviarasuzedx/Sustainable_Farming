from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model  # type: ignore
import io
import base64
from flask_sqlalchemy import SQLAlchemy
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# ─── Database Configurations ─────────────────────
# Profit Prediction Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'  # Main database for profit predictions
app.config['SQLALCHEMY_BINDS'] = {
    'crops': 'sqlite:///crop_predictions.db'  # Separate database for crop recommendations
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ─── AI Chat Assistant Config ────────────────────
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-7b-instruct"

chat_headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ─── Profit Prediction Models ───────────────────
try:
    profit_model = load_model("crop_profit_model.h5")
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("numeric_cols.pkl", "rb") as f:
        numeric_cols = pickle.load(f)
    with open("kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
except Exception as e:
    print(f"Error loading profit prediction models: {e}")

# ─── Crop Recommendation Models ─────────────────
try:
    crop_model = joblib.load("crop_predictor_xgb_full_model.pkl")
    crop_encoder = joblib.load("label_encoder.pkl")
    feature_means = joblib.load("feature_means.pkl")
except Exception as e:
    print(f"Error loading crop recommendation models: {e}")

# ─── Database Models ────────────────────────────

# Profit Prediction Model
class ProfitPrediction(db.Model):
    __bind_key__ = None  # Uses default database
    id = db.Column(db.Integer, primary_key=True)
    crop = db.Column(db.String(50))
    competitor_price = db.Column(db.Float)
    demand_index = db.Column(db.Float)
    supply_index = db.Column(db.Float)
    economic_indicator = db.Column(db.Float)
    weather_impact = db.Column(db.Float)
    seasonal_factor = db.Column(db.Float)
    consumer_trend = db.Column(db.Float)
    profit_probability = db.Column(db.Float)
    status = db.Column(db.String(20))

# Crop Recommendation Model
class CropPrediction(db.Model):
    __bind_key__ = 'crops'  # Uses the crops database
    __tablename__ = 'crop_predictions'
    id = db.Column(db.Integer, primary_key=True)
    soil_ph = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    fertilizer = db.Column(db.Float, nullable=False)
    pesticide = db.Column(db.Float, nullable=False)
    predicted_crop = db.Column(db.String(50), nullable=False)
    prediction_time = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<CropPrediction {self.id} - {self.predicted_crop}>'

# ─── Routes ─────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

# ─── Profit Prediction Routes ───────────────────
@app.route("/Farm_advise", methods=["GET"])
def Farm_adviser():
    return render_template("Farm_adviser.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        crop_name = request.form.get("crop", "").capitalize()
        competitor_price = float(request.form.get("competitor_price", 0))
        demand_index = float(request.form.get("demand_index", 0))
        supply_index = float(request.form.get("supply_index", 0))
        economic_indicator = float(request.form.get("economic_indicator", 0))
        weather_impact_score = float(request.form.get("weather_impact_score", 0))
        seasonal_factor = float(request.form.get("seasonal_factor", 0))
        consumer_trend_index = float(request.form.get("consumer_trend_index", 0))

        sample_data = pd.DataFrame([{
            'Product': crop_name,
            'competitor_price': competitor_price,
            'demand_index': demand_index,
            'supply_index': supply_index,
            'economic_indicator': economic_indicator,
            'weather_impact': weather_impact_score,
            'seasonal_factor': seasonal_factor,
            'consumer_trend': consumer_trend_index
        }])

        if crop_name in label_encoders['Product'].classes_:
            sample_data['Product'] = label_encoders['Product'].transform([crop_name])[0]
        else:
            sample_data['Product'] = 0

        numeric_input = sample_data.copy()
        for col in set(numeric_cols) - set(numeric_input.columns):
            numeric_input[col] = 0.0
        numeric_input = numeric_input[numeric_cols]
        numeric_input[numeric_cols] = scaler.transform(numeric_input[numeric_cols])

        cluster_label = kmeans.predict(numeric_input)[0]
        sample_data['Cluster_Label'] = cluster_label
        features = numeric_input.copy()
        features['Cluster_Label'] = cluster_label

        crop_encoded = np.array([[sample_data['Product'].values[0]]])
        model_inputs = [features, crop_encoded]
        profit_probability = profit_model.predict(model_inputs)[0][0]
        profit_status = "Profitable ✅" if profit_probability > 0.5 else "Not Profitable ❌"

        # Save to DB
        new_prediction = ProfitPrediction(
            crop=crop_name,
            competitor_price=competitor_price,
            demand_index=demand_index,
            supply_index=supply_index,
            economic_indicator=economic_indicator,
            weather_impact=weather_impact_score,
            seasonal_factor=seasonal_factor,
            consumer_trend=consumer_trend_index,
            profit_probability=float(profit_probability),
            status=profit_status
        )
        db.session.add(new_prediction)
        db.session.commit()

        # Chart
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ["Not Profitable", "Profitable"]
        values = [1 - profit_probability, profit_probability]
        colors = ["red", "green"]
        bars = ax.bar(labels, values, color=colors)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{value:.2f}", ha='center')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Profitability Prediction")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        chart_url = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        chart_url = f"data:image/png;base64,{chart_url}"

        return render_template("Farm_adviser.html", status=profit_status, probability=round(profit_probability, 2), chart_url=chart_url)

    except Exception as e:
        return f"Error: {e}"
#________________________________________________________________________________
#@app.route("/history1")
#def history1():
#    all_predictions = ProfitPrediction.query.order_by(ProfitPrediction.id.desc()).all()
 #   return render_template("history1.html", new_predictions=all_predictions)

# ─── Crop Recommendation Routes ─────────────────
@app.route("/crop_adviser")
def crop_adviser():
    return render_template("crop_adviser.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.route("/Weather_Irrigation")
def Weather_Irrigation():
    return render_template("Weather_Irrigation.html")

@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    data = request.get_json()

    soil_ph = float(data['soil_ph'])
    temp_c = float(data['temperature'])
    rainfall = float(data['rainfall'])
    fertilizer = float(data['fertilizer'])
    pesticide = float(data['pesticide'])

    features = [[
        soil_ph,
        feature_means['Soil_Moisture'],
        temp_c,
        rainfall,
        fertilizer,
        pesticide,
        feature_means['Crop_Yield_ton'],
        feature_means['Sustainability_Score']
    ]]

    prediction_proba = crop_model.predict_proba(features)[0]
    predicted_index = np.argmax(prediction_proba)
    predicted_crop = crop_encoder.inverse_transform([predicted_index])[0]
    class_labels = list(crop_encoder.classes_)

    # Save to DB
    new_entry = CropPrediction(
        soil_ph=soil_ph,
        temperature=temp_c,
        rainfall=rainfall,
        fertilizer=fertilizer,
        pesticide=pesticide,
        predicted_crop=predicted_crop
    )
    db.session.add(new_entry)
    db.session.commit()

    return jsonify({
        'crop': predicted_crop,
        'probabilities': prediction_proba.tolist(),
        'labels': class_labels
    })

@app.route('/history')
def history():
    # Fetch from both models
    crop_predictions = CropPrediction.query.order_by(CropPrediction.prediction_time.desc()).all()
    profit_predictions = ProfitPrediction.query.order_by(ProfitPrediction.id.desc()).all()

    return render_template('history.html', 
                           crop_predictions=crop_predictions, 
                           profit_predictions=profit_predictions)


# ─── Chatbot Routes ─────────────────────────────
@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask():
    messages = request.json.get("messages", [])

    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=chat_headers, json=data)
        result = response.json()

        if 'error' in result:
            return jsonify({"error": result['error']['message']})
        else:
            return jsonify({"response": result['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify({"error": str(e)})
    


# ─── Main ────────────────────────────────────────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # No need for 'bind'
    app.run(debug=True)