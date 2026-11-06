import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Healthcare Analytics Dashboard", page_icon="🏥", layout="wide")
OUTPUT_DIR = "project_outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "readmission_model.joblib")
DEMO_PATH = os.path.join(OUTPUT_DIR, "demo_patient_data.csv")

st.title("🏥 Web-Based Healthcare Analytics")
st.caption("Healthcare Analytics • Graph Analytics • New Patient Readmission Detection")

if not os.path.exists(MODEL_PATH):
    st.error("Run the Colab notebook first so the saved Random Forest model is available.")
    st.stop()

model = joblib.load(MODEL_PATH)
demo = pd.read_csv(DEMO_PATH) if os.path.exists(DEMO_PATH) else pd.DataFrame()

page = st.sidebar.radio("Navigation", ["Healthcare Analytics", "Graph Analytics", "New Patient Readmission Detection"])

if page == "Healthcare Analytics":
    st.header("📊 Healthcare Analytics")
    if demo.empty:
        st.warning("Run the notebook first to generate the dashboard data.")
        st.stop()
    tabs = st.tabs(["Age Analysis", "Hospital Stay", "Previous Visits"])
    with tabs[0]:
        age_rate = demo.groupby("age")["readmission"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(age_rate.index, age_rate.values)
        ax.set_title("Readmission Rate by Age Group")
        ax.set_ylabel("Readmission Rate (%)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
    with tabs[1]:
        stay_rate = demo.groupby("time_in_hospital")["readmission"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(stay_rate.index, stay_rate.values, marker="o")
        ax.set_title("Readmission Rate vs Hospital Stay")
        ax.set_xlabel("Days in Hospital")
        ax.set_ylabel("Readmission Rate (%)")
        fig.tight_layout()
        st.pyplot(fig)
    with tabs[2]:
        visit_rate = demo.groupby("number_inpatient")["readmission"].mean().mul(100)
        visit_rate = visit_rate[visit_rate.index <= 8]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(visit_rate.index, visit_rate.values, marker="o")
        ax.set_title("Readmission Rate vs Previous Inpatient Visits")
        ax.set_xlabel("Previous Inpatient Visits")
        ax.set_ylabel("Readmission Rate (%)")
        fig.tight_layout()
        st.pyplot(fig)

elif page == "Graph Analytics":
    st.header("🕸️ Graph Analytics")
    st.write("Patient–Diagnosis Network with Degree Centrality, Betweenness Centrality, and PageRank.")
    graph_path = os.path.join(OUTPUT_DIR, "05_patient_diagnosis_network.png")
    pagerank_path = os.path.join(OUTPUT_DIR, "06_top_diagnoses_pagerank.png")
    if os.path.exists(graph_path):
        st.image(graph_path, caption="Patient–Diagnosis Network", use_container_width=True)
    if os.path.exists(pagerank_path):
        st.image(pagerank_path, caption="Top Diagnoses by PageRank", use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Degree Centrality", "Connections")
    c2.metric("Betweenness Centrality", "Bridge role")
    c3.metric("PageRank", "Network importance")

else:
    st.header("🧑‍⚕️ New Patient Readmission Detection")
    st.write("Enter details for a new patient and click the button to obtain a model prediction.")
    c1, c2 = st.columns(2)
    with c1:
        age = st.selectbox("Age Group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=7)
        gender = st.selectbox("Gender", ["Male", "Female"])
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
        num_medications = st.slider("Number of Medications", 0, 80, 15)
    with c2:
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 6)
        number_outpatient = st.slider("Previous Outpatient Visits", 0, 10, 0)
        number_emergency = st.slider("Previous Emergency Visits", 0, 10, 0)
        number_inpatient = st.slider("Previous Inpatient Visits", 0, 10, 0)

    if st.button("🔍 Detect Readmission Risk", type="primary"):
        new_patient = pd.DataFrame([{
            "age": age, "gender": gender, "time_in_hospital": time_in_hospital,
            "num_medications": num_medications, "number_diagnoses": number_diagnoses,
            "number_outpatient": number_outpatient, "number_emergency": number_emergency,
            "number_inpatient": number_inpatient
        }])
        prediction = model.predict(new_patient)[0]
        probability = model.predict_proba(new_patient)[0, 1]
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ Predicted: Readmission")
        else:
            st.success("✅ Predicted: Not Readmitted")
        st.metric("Estimated Readmission Probability", f"{probability:.2%}")
        st.progress(float(probability))
        st.info("Academic project demonstration only. This is not a clinical diagnosis.")
