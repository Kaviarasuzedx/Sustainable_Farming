
# 🌾 FarmAssist - Smart Crop Recommendation & Profitability Prediction

FarmAssist is a Flask-based web application that helps farmers and agriculture enthusiasts make smart decisions about crop selection and profitability. It uses machine learning models to recommend crops based on soil and weather features, and also predicts potential profitability.



Data-Driven AI for Sustainable Farming

Challenge Overview:
Agriculture plays a vital role in sustaining life, but its environmental and economic impact is substantial. With growing challenges like water scarcity, excessive pesticide use, and soil degradation, the need for more sustainable agricultural practices has never been greater. This hackathon aims to leverage AI technologies to create innovative solutions that promote sustainability, optimize resource usage, and improve the livelihoods of farmers. 

Develop a multi-agentic AI system that brings together different stakeholders in agriculture—farmers, weather stations, and agricultural experts—to work collaboratively for the optimization of farming practices. 

The goals is to reduce environmental impact of farming: Promote practices that lower the carbon footprint, minimize water consumption, and reduce soil erosion.

Current Process:
Farmer Advisor: Provides actionable insights by analyzing input from the farmer about land, crop preferences, and financial goals.

Market Researcher : Analyzes regional market trends, crop pricing, and demand forecasts to suggest the most profitable crops to plant.

Expected Technical Output: Multiagent framework and SQLite Database for long term memory



---

## 🗂️ Project Structure

```
Flask-FarmAdviso/
├── app.py                            # Main Flask application
├── templates/
│   ├──  index.html                   # Frontend HTML for home page to navigate
│   ├──  crop_adviser.html            # crop predicts
│   ├──  Farm_adviser.html            # predicts profit or lose
│   ├──  history.html                 # show the history crop predicts and predicts profit or lose
│   ├──  chatbot.html                 # ai chart 
│   ├──  team.html                    # contain team information
│   └──  Weather_Irrigation.html      # wheather checking and tips for saving water     
├── static/
│   ├── static/
│   │   └──images                      # images for team profiles
│   ├── style.css                     # CSS styling for frontend
│   └── script.js                     # (Optional) JS for UI interactivity
├── crop_predictor_xgb_full_model.pkl    # Trained XGBoost model for crop recommendation
├── crop_profit_model.h5             # TensorFlow model for profitability prediction
├── scaler.pkl                       # StandardScaler for input normalization
├── label_encoder.pkl                # LabelEncoder for encoding categorical labels
├── label_encoders.pkl               # Encoders for multiple features
├── kmeans.pkl                       # KMeans clustering model
├── feature_means.pkl                # Mean values for features used in preprocessing
├── region_encoder.pkl               # Encoder for regional data
├── numeric_cols.pkl                 # Pickle file containing numeric column names
└── requirements.txt                 # List of Python dependencies
```

---

## 🚀 Features

- 🌱 Crop recommendation based on soil pH, temperature, rainfall, humidity, and sunlight.
- 💰 Crop profitability prediction using TensorFlow.
- 🔍 Region-aware model with clustering.
- 🧠 Scaled and encoded input processing using pre-trained encoders.
- 🧾 User-friendly web interface with HTML/CSS.

---

## 💻 Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript (optional)
- **ML Models:** XGBoost, TensorFlow (Keras)
- **Others:** Pickle, scikit-learn

---

## 🔧 Installation and Setup

1. **Clone the repository**
```bash
git clone https://github.com/Kaviarasuzedx/Sustainable_Farming.git
cd Flask-FarmAdviso
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Flask app**
```bash
python app.py
```

4. Open your browser and visit:
```
http://127.0.0.1:5000
```

---

## 📊 Model Descriptions

### ✅ Crop Recommendation Model (`crop_predictor_xgb_full_model.pkl`)
- Algorithm: XGBoost
- Input Features: Soil pH, Rainfall, Humidity, Temperature, Sunlight
- Output: Predicted crop name

### 📈 Profitability Prediction Model (`crop_profit_model.h5`)
- Algorithm: Deep Neural Network (Keras)
- Input Features: Numerical and categorical crop data (with encoding)
- Output: Expected profit or profitability score

---

## 🧪 Sample Input

```json
{
  "soil_ph": 6.5,
  "temperature": 28,
  "humidity": 70,
  "rainfall": 200,
  "sunlight_hour": 8,
  "region": "Tamil Nadu"
}
```

---

## 🖼️ Screenshots (Add your actual images here)
| Input Form | Result |
|------------|--------|
| ![form](screenshots/screenshot1.png) | ![result](screenshots/screenshot2.png) | ![result](screenshots/screenshot3.png) |  ![result](screenshots/screenshot4.png) |
.

---

## 🙌 Credits

Created by **Kaviarasu**, **Sriram** 
Feel free to contribute or report issues!

---

## 📬 Contact

If you have any feedback or want to collaborate:

- GitHub: [https://github.com/Kaviarasuzedx]
- Email: [kaviarasu6380@gmail.com]

- GitHub: [https://github.com/Sriramking2]
- Email: [srirampurushothaman2004@gmail.com]



