import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LOAD MODEL AND SCALER ---
try:
    rf_model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('heart_scaler.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'heart_disease_model.pkl' and 'heart_scaler.pkl' are in the correct directory.")
    st.stop()


# --- EXPECTED COLUMNS ---
expected_columns = [
    'id', 'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca',
    'sex_Female', 'sex_Male', 'dataset_Cleveland', 'dataset_Hungary',
    'dataset_Switzerland', 'dataset_VA Long Beach', 'cp_asymptomatic',
    'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'fbs_False', 'fbs_True', 'restecg_lv hypertrophy', 'restecg_normal',
    'restecg_st-t abnormality', 'exang_False', 'exang_True',
    'slope_downsloping', 'slope_flat', 'slope_upsloping',
    'thal_fixed defect', 'thal_normal', 'thal_reversable defect'
]

# --- HELPER FUNCTIONS ---
def preprocess_input(data):
    """Preprocesses user input to match the model's training data format."""
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=expected_columns, fill_value=0)
    scaled_features = scaler.transform(df)
    return scaled_features

def create_gauge_chart(probability):
    """Creates a gauge chart for the prediction probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk", 'font': {'size': 24, 'color': '#2c3e50'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#3498db"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#27ae60'},
                {'range': [50, 75], 'color': '#f39c12'},
                {'range': [75, 100], 'color': '#e74c3c'}],
            }))
    fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'family': "Arial"})
    return fig

def create_bp_chart(trestbps):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = trestbps,
        title = {'text': "Resting Blood Pressure", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [90, 200]},
            'steps': [
                {'range': [90, 120], 'color': 'green'},
                {'range': [120, 140], 'color': 'orange'},
                {'range': [140, 200], 'color': 'red'}],
            'bar': {'color': 'black', 'thickness': 0.2}
        }))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def create_chol_chart(chol):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = chol,
        title = {'text': "Cholesterol", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [100, 400]},
            'steps': [
                {'range': [100, 200], 'color': 'green'},
                {'range': [200, 240], 'color': 'orange'},
                {'range': [240, 400], 'color': 'red'}],
            'bar': {'color': 'black', 'thickness': 0.2}
        }))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    return fig
    
def create_hr_chart(thalch, age):
    max_hr_pred = 220 - age
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = thalch,
        title = {'text': "Max Heart Rate", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [60, 220]},
            'steps': [
                {'range': [60, max_hr_pred - 20], 'color': 'red'},
                {'range': [max_hr_pred - 20, max_hr_pred], 'color': 'green'},
                {'range': [max_hr_pred, 220], 'color': 'orange'}],
            'bar': {'color': 'black', 'thickness': 0.2},
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.9, 'value': max_hr_pred}
        }))
    fig.add_annotation(x=0.5, y=0.15, text=f"Age-Predicted Max: {max_hr_pred}", showarrow=False, font=dict(size=12))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=30))
    return fig


# --- STYLES ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        color: white;
        background-color: #007bff;
        border-radius:12px;
        font-size:16px;
        font-weight: bold;
        padding:10px 24px;
        border: none;
        transition-duration: 0.4s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
    }
    .metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.1);
        color: white;
        margin-bottom: 10px;
    }
    .metric-box h3 {
        font-size: 1rem;
        font-weight: 300;
        margin-bottom: 5px;
    }
    .metric-box p {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .box-1 { background: linear-gradient(135deg, #FF6B6B, #FFC371); }
    .box-2 { background: linear-gradient(135deg, #6B8BFF, #A1C4FD); }
    .box-3 { background: linear-gradient(135deg, #42E695, #38F9D7); }
    .box-4 { background: linear-gradient(135deg, #8E2DE2, #4A00E0); }
    .box-5 { background: linear-gradient(135deg, #FF9A8B, #FF6A88); }
    .box-6 { background: linear-gradient(135deg, #00C9FF, #92FE9D); }
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("Enter patient details in the sidebar to get a real-time prediction of heart disease risk and contextual health insights.")

# --- SIDEBAR FOR INPUTS ---
st.sidebar.header("👨‍⚕️ Patient Data Input")
with st.sidebar:
    age = st.slider("Age", 20, 100, 50, help="Patient's age in years.")
    sex = st.radio("Sex", ['Male', 'Female'], help="Patient's gender.")
    cp = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'], help="Type of chest pain experienced.")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [True, False])
    restecg = st.selectbox("Resting ECG Results", ['normal', 'st-t abnormality', 'lv hypertrophy'])
    thalch = st.slider("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.radio("Exercise Induced Angina", [True, False])
    oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ['upsloping', 'flat', 'downsloping'])
    ca = st.select_slider("Major Vessels Colored by Flouroscopy", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ['normal', 'fixed defect', 'reversable defect'])

    predict_button = st.button("📈 Predict Heart Disease Risk")

# --- MAIN DASHBOARD AREA ---
col1, col2, col3 = st.columns([1, 1, 1])

# Column 1: Display user inputs in colored boxes
with col1:
    st.subheader("📊 Patient Vitals")
    st.markdown(f'<div class="metric-box box-1"><h3>Age</h3><p>{age} yrs</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box box-2"><h3>Sex</h3><p>{sex}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box box-3"><h3>Chest Pain</h3><p>{cp.replace("_", " ").title()}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box box-4"><h3>Resting BP</h3><p>{trestbps} mm Hg</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box box-5"><h3>Cholesterol</h3><p>{chol} mg/dl</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-box box-6"><h3>Max HR</h3><p>{thalch} bpm</p></div>', unsafe_allow_html=True)

# Column 2: Contextual Health Graphs
with col2:
    st.subheader("📈 Health Metrics Analysis")
    st.plotly_chart(create_bp_chart(trestbps), use_container_width=True)
    st.plotly_chart(create_chol_chart(chol), use_container_width=True)
    st.plotly_chart(create_hr_chart(thalch, age), use_container_width=True)


# Column 3: Prediction Analysis
with col3:
    st.subheader("🩺 Prediction Analysis")
    if predict_button:
        with st.spinner("Analyzing data..."):
            input_data = {
                'id': 1, 'age': age, 'sex': sex, 'dataset': 'Cleveland', 'cp': cp,
                'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg,
                'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
                'ca': ca, 'thal': thal
            }
            processed_input = preprocess_input(input_data)
            prediction = rf_model.predict(processed_input)
            prediction_proba = rf_model.predict_proba(processed_input)[0][1]

            st.plotly_chart(create_gauge_chart(prediction_proba), use_container_width=True)

            if prediction[0] == 1:
                st.error(f"**High Risk:** The model predicts a significant probability of heart disease.", icon="⚠️")
            else:
                st.success(f"**Low Risk:** The model predicts a low probability of heart disease.", icon="✅")
    else:
        st.info("Click the predict button to see the analysis.", icon="ℹ️")


# --- METRIC EXPLANATIONS ---
with st.expander("Learn more about the metrics and graphs"):
    st.markdown("""
    - **Patient Vitals:** A summary of the data you entered in the sidebar.
    - **Health Metrics Analysis:** These charts compare your input values to general health guidelines:
        - **Resting Blood Pressure:** Green is Normal (<120), Orange is Elevated (120-139), Red is High (>140).
        - **Cholesterol:** Green is Desirable (<200), Orange is Borderline High (200-239), Red is High (>240).
        - **Max Heart Rate:** Shows your achieved rate against the age-predicted maximum (220 - Age). Being close to the target during exercise is generally good.
    - **Prediction Analysis:** The gauge shows the model's confidence in its prediction. A higher percentage indicates a greater predicted risk of heart disease.
    """)

