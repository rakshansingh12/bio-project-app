import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("🧬 Disease Prediction App")

df = pd.read_csv("final_selected_dataset.csv")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.header("Input Features")

input_data = []

for col in X.columns:
    val = st.sidebar.slider(
        col,
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    input_data.append(val)

input_df = pd.DataFrame([input_data], columns=X.columns)

prediction = model.predict(input_df)[0]

st.subheader("Prediction Result")

if prediction == 1:
    st.error("⚠️ High Risk Detected")
else:
    st.success("✅ Low Risk")
