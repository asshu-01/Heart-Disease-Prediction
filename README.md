# ❤️ Heart Disease Prediction Dashboard

A machine learning–powered **Heart Disease Risk Prediction Web App** built using **Streamlit, Random Forest**, and real-world medical dataset features.  
This app predicts the likelihood of heart disease based on patient clinical data and visualizes risk through interactive charts.

---

## 🚀 Features

- 🔹 Real-time heart disease risk prediction  
- 🔹 Gauge visualization for risk confidence  
- 🔹 Health metric dashboards:
  - Resting Blood Pressure Indicator
  - Cholesterol Indicator
  - Max Heart Rate Comparison (Age-predicted)
- 🔹 Clean, modern UI with sidebar-based inputs
- 🔹 Fully interactive Plotly graphs

---

## 🧠 Machine Learning Model

- Algorithm: **Random Forest Classifier**
- Data Preprocessing: One-hot encoding + standard scaling
- Trained on heart disease dataset with engineered features

Model & Scaler files:
   heart_disease_model.pkl
   heart_scaler.pkl

   
---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| UI | Streamlit |
| Backend | Python |
| ML | Scikit-learn |
| Visualization | Plotly |
| Packaging | Joblib |

---

## 📥 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd <your-project-folder>
2️⃣ Create & activate virtual environment
python -m venv venv
venv\Scripts\activate  # For Windows
3️⃣ Install dependencies
pip install -r requirements.txt

If requirements.txt doesn't exist, run:

pip install streamlit pandas numpy scikit-learn joblib plotly

▶️ Run the Application
streamlit run app.py




Open the local URL shown in terminal to access the app.

🧩 Input Fields

The app accepts the following clinical inputs:

Age

Sex

Chest Pain Type

Resting Blood Pressure

Serum Cholesterol

Fasting Blood Sugar

Resting ECG

Max Heart Rate Achieved

Exercise Induced Angina

ST Depression

Slope of ST Segment

Major Vessels (0–3)

Thalassemia Types

These features are automatically preprocessed to match the model’s format.

📊 Output Interpretation

Gauge chart shows risk probability (%)

Color coding:

🟢 0–50% → Low Risk

🟠 50–75% → Moderate Risk

🔴 75–100% → High Risk

A diagnosis message displays accordingly.

📁 Project Structure
📂 project-folder
│── app.py
│── heart_disease_model.pkl
│── heart_scaler.pkl
│── heart_disease_uci.csv (optional)
│── README.md

🧑‍⚕️ Disclaimer

This tool is for educational purposes only
Not to be used for medical diagnosis or treatment decisions.

🔮 Future Enhancements

SHAP explainability for model insights

Multi-dataset training support

Cloud deployment options

Save prediction history
