import streamlit as st
import pandas as pd
import numpy as np
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
# LOAD DATA
# -----------------------------
df = pd.read_csv("final_selected_dataset.csv")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# INPUT SECTION (REAL VALUES)
# -----------------------------
st.sidebar.title("🔧 Patient Information")

input_data = {}

# Real-world values (fix for negative issue)
input_data["Age"] = st.sidebar.slider("Age", 10, 80, 30)
input_data["BloodPressure"] = st.sidebar.slider("Blood Pressure", 50, 180, 100)
input_data["SkinThickness"] = st.sidebar.slider("Skin Thickness", 10, 100, 30)

# Add remaining features automatically
for col in X.columns:
    if col not in input_data:
        input_data[col] = st.sidebar.slider(col, 0.0, 1.0, 0.5)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
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
    else:
        st.success("✅ Low Risk")

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
# EXPLANATION SECTION
# -----------------------------
st.markdown("---")

st.subheader("🔍 About the Model")

st.markdown("""
This system uses:

- **Adaptive Feature Selection**
- **Biological Relevance Scoring**
- **Random Forest Machine Learning**

to improve prediction accuracy and interpretability.
""")
