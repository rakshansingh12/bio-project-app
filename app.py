import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Bio Feature Selection System", layout="wide")

st.title("🧬 Intelligent Disease Prediction System")
st.markdown("Combining **Statistical Feature Selection + Biological Relevance**")

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
# SIDEBAR INPUT (REAL VALUES)
# -----------------------------
st.sidebar.header("🔧 Patient Information")

input_data = {}

# Define realistic ranges manually
input_data["Age"] = st.sidebar.slider("Age", 10, 80, 30)
input_data["BloodPressure"] = st.sidebar.slider("Blood Pressure", 50, 180, 100)
input_data["SkinThickness"] = st.sidebar.slider("Skin Thickness", 10, 100, 30)

# Add other features if present
for col in X.columns:
    if col not in input_data:
        input_data[col] = st.sidebar.slider(col, 0.0, 1.0, 0.5)

input_df = pd.DataFrame([input_data])

# -----------------------------
# NORMALIZE INPUT (IMPORTANT)
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
# OUTPUT SECTION
# -----------------------------
st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error("⚠️ High Risk Detected")
    else:
        st.success("✅ Low Risk")

with col2:
    st.metric("Confidence", f"{probability*100:.2f}%")

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
# INPUT SUMMARY
# -----------------------------
st.subheader("📥 Input Summary")
st.write(input_df)

# -----------------------------
# ABOUT SECTION
# -----------------------------
st.markdown("---")
st.markdown("""
### 🔍 About This System
This system uses:
- Adaptive Variance Feature Selection
- Biological Relevance Scoring
- Machine Learning (Random Forest)

to improve prediction accuracy and interpretability.
""")
