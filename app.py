import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# -----------------------------
# CUSTOM STYLE
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #00FFAA;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🧬 Intelligent Disease Prediction System")
st.markdown("### AI-powered feature selection + prediction")
st.markdown("---")

# -----------------------------
# LOAD DATA (CACHED)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_selected_dataset.csv")

df = load_data()

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# -----------------------------
# SCALING + MODEL (CACHED)
# -----------------------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.sidebar.title("🔧 Patient Information")

# UI → Dataset mapping
feature_mapping = {
    "Age": "Age",
    "Blood Pressure": "BloodPressure",
    "Skin Thickness": "SkinThickness"
}

# User input (UI friendly)
input_data_ui = {}

input_data_ui["Age"] = st.sidebar.slider("Age", -3.0, 3.0, 0.0)
input_data_ui["Blood Pressure"] = st.sidebar.slider("Blood Pressure", -3.0, 3.0, 0.0)
input_data_ui["Skin Thickness"] = st.sidebar.slider("Skin Thickness", -3.0, 3.0, 0.0)

# Convert UI → model format
input_data = {}

for ui_name, value in input_data_ui.items():
    actual_col = feature_mapping[ui_name]
    input_data[actual_col] = value

# Fill remaining features with mean
import random

for col in X.columns:
    if col not in input_data:
        input_data[col] = float(X[col].mean()) + random.uniform(-0.5, 0.5)

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]

# -----------------------------
# SCALE INPUT
# -----------------------------
input_scaled = scaler.transform(input_df)

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col1, col2 = st.columns(2)

# LEFT: RESULT
with col1:
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk Detected")
        st.write("The model predicts higher risk based on the input features.")
    else:
        st.success("✅ Low Risk")
        st.write("The model predicts lower risk based on the input features.")

    st.metric("Confidence Score", f"{probability*100:.2f}%")
    st.progress(int(probability * 100))

# RIGHT: INPUT SUMMARY
with col2:
    st.subheader("📥 Input Summary")
    st.dataframe(input_df)

st.markdown("---")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("🧠 Feature Importance")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(feature_importance.set_index("Feature"))

# -----------------------------
# ABOUT SECTION
# -----------------------------
st.markdown("---")

st.subheader("🔍 About the Model")

st.markdown("""
This system uses:

- Adaptive Feature Selection  
- Biological Relevance Scoring  
- Random Forest Machine Learning  

to improve prediction accuracy and interpretability.
""")
