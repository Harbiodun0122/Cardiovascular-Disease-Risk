import joblib
import pandas as pd
import streamlit as st

# Load model and pipeline
model = joblib.load(r'C:/Users/Harbiodun/Downloads/Assignment/Success project/MLP-CVD.pkl')
preprocessing_pipeline = joblib.load(r'C:/Users/Harbiodun/Downloads/Assignment/Success project/pipeline.pkl')

# columns order
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
final_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

# Configure page
st.set_page_config(
    page_title="CVD Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Sidebar: image + text only
st.sidebar.image("C:/Users/Harbiodun/Downloads/Assignment/Success project/pic.png", use_column_width=True)
st.sidebar.markdown("""
### 💡 Stay Heart Smart

Take control of your heart health by understanding your risk early.

*“An ounce of prevention is worth a pound of cure.”*
""")

# Custom top bar
st.markdown("""
    <style>
    .top-bar {
        background-color: #004d7a;
        padding: 20px 10px;
        color: white;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
    }
    .top-bar h1 {
        color: white;
        margin: 0;
        font-size: 2.4em;
    }
    .top-bar p {
        margin: 5px 0 0;
        font-size: 1.1em;
    }
    </style>
    <div class="top-bar">
        <h1>Cardiovascular Disease Risk Prediction</h1>
        <p>Enter patient data below to assess their cardiovascular risk.</p>
    </div>
""", unsafe_allow_html=True)


# Main input area
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("**Age**", min_value=1, max_value=150)
    sex = st.selectbox("**Sex**", ["Male", "Female"])
    cp = st.selectbox("**Chest Pain Type**", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
    trestbps = st.number_input("**Resting Blood Pressure**", value=120)
    thal = st.selectbox("**Thalassemia**", ["normal", "fixed defect", "reversable defect"])

with col2:
    chol = st.number_input("**Cholesterol**", value=200)
    fbs = st.selectbox("**Fasting Blood Sugar > 120 mg/dl**", [True, False])
    restecg = st.selectbox("**Resting ECG**", ["normal", "st-t wave abnormality", "lv hypertrophy"])
    thalch = st.number_input("**Max Heart Rate**", value=150)

with col3:
    exang = st.selectbox("**Exercise-Induced Angina**", [True, False])
    oldpeak = st.number_input("**ST Depression**", value=1.0)
    slope = st.selectbox("**Slope of ST**", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("**Number of major vessels (0-3)**", min_value=0.0, max_value=3.0, step=1.0)
    

# Prediction button
if st.button("Predict Risk"):

    new_predictions = [age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]

    new_predictions = pd.DataFrame(new_predictions, index=columns).T
    
    pipeline = preprocessing_pipeline.transform(new_predictions)
        
    pipeline_df = pd.DataFrame(pipeline, columns=final_cols)

    prediction = model.predict(pipeline_df)

    result = prediction[0]
    
    # if result == 1:
    #     st.error("High risk of heart disease.")
    # else:
    #     st.success("Low risk of heart disease.")

   
    if prediction == 1:
        st.error("⚠️ The patient is at **high risk** of cardiovascular disease.")
    else:
        st.success("✅ The patient is at **low risk** of cardiovascular disease.")


st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #004d7a;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 0.9em;
    }
    </style>

    <div class="footer">
        © 2025 Osidele Success Final Year Project
    </div>
""", unsafe_allow_html=True)        