# Web-Based Healthcare Analytics and Patient Readmission Risk Prediction

## Project Overview

This project combines healthcare data analytics, graph analytics, machine learning, and web technologies to analyze hospital patient records and predict patient readmission risk.

The system uses the **Diabetes 130-US Hospitals dataset** from the UCI Machine Learning Repository, containing 101,766 hospital encounters collected from 130 US hospitals between 1999 and 2008.

The project analyzes patient readmission patterns, constructs a patient–diagnosis network, identifies important diagnoses using graph centrality measures and PageRank, and predicts readmission risk using machine learning models.

A Streamlit-based web dashboard provides interactive healthcare analytics, graph visualizations, and new patient readmission predictions.

## Objectives

- Analyze hospital patient records to identify readmission patterns.
- Perform data cleaning and preprocessing on healthcare data.
- Build a patient–diagnosis network using NetworkX.
- Analyze graph structures using Degree Centrality, Betweenness Centrality, and PageRank.
- Identify important diagnoses based on their network relationships.
- Predict patient readmission risk using Logistic Regression and Random Forest.
- Compare machine learning models using evaluation metrics.
- Develop a Streamlit web dashboard for healthcare analytics and new patient prediction.

## Technologies Used

- **Programming Language:** Python
- **Platform:** Google Colab / Jupyter Notebook
- **Data Analysis:** Pandas, NumPy
- **Graph Analytics:** NetworkX
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib
- **Web Application:** Streamlit
- **Public Web Access:** pyngrok

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

Dataset details:

- 101,766 hospital encounters.
- Data collected from 130 US hospitals.
- Covers ten years of clinical care (1999–2008).
- Contains patient demographic, admission, diagnosis, medication, and hospital stay information.

**Dataset Source:** [UCI Machine Learning Repository – Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

The dataset is loaded directly using the `ucimlrepo` Python package.

## Methodology

### 1. Data Loading and Preprocessing

- Load the healthcare dataset from the UCI repository.
- Replace missing-value indicators (`?`) with null values.
- Remove unnecessary identifier and high-missingness columns.
- Convert the original readmission category into a binary target:
  - 0 – Not Readmitted
  - 1 – Readmitted (`<30` or `>30`)

### 2. Healthcare Data Analytics

Analyze readmission patterns based on:

- Patient age groups.
- Previous inpatient visits.
- Length of hospital stay.

Generate visualizations to understand how these factors relate to readmission.

### 3. Patient–Diagnosis Graph Construction

Build an undirected patient–diagnosis network using NetworkX.

- Use a reproducible sample of 5,000 hospital encounters.
- Represent patients and diagnoses as separate node types.
- Connect each patient to their primary, secondary, and tertiary diagnoses.
- Identify frequently shared diagnoses as important network hubs.

### 4. Graph Analytics

Apply three graph analysis techniques:

- **Degree Centrality:** Measures the number of connections of a node relative to the network.
- **Betweenness Centrality:** Identifies nodes that act as bridges along shortest paths.
- **PageRank:** Identifies important nodes based on the importance of their connected nodes.

Identify and rank important diagnosis nodes using these metrics.

### 5. Machine Learning

Train and compare two classification models:

**Logistic Regression**
- Used as a baseline classification model.
- Configured with balanced class weights.

**Random Forest**
- Uses 200 decision trees.
- Configured with balanced class weights.
- Used for new patient readmission prediction in the dashboard.

Selected input features:

- Age
- Gender
- Time in hospital
- Number of medications
- Number of diagnoses
- Previous outpatient visits
- Previous emergency visits
- Previous inpatient visits

The dataset is divided into 80% training data and 20% testing data using stratified sampling.

### 6. Model Evaluation

Evaluate both models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Reported test-set results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.6222 | 0.5699 |
| Precision | 0.6183 | 0.5334 |
| Recall | 0.4711 | 0.5343 |
| F1 Score | 0.5347 | 0.5338 |
| ROC-AUC | 0.6509 | 0.5914 |

Logistic Regression performs better on four of the five evaluation metrics, while Random Forest achieves higher recall.

The Random Forest pipeline is saved using Joblib and reused in the web dashboard.

### 7. Web-Based Healthcare Analytics Dashboard

Develop an interactive dashboard using Streamlit with three main sections:

**Healthcare Analytics**
- Readmission analysis by age group.
- Readmission analysis based on hospital stay.
- Readmission analysis based on previous inpatient visits.

**Graph Analytics**
- Patient–diagnosis network visualization.
- Top diagnoses ranked using PageRank.
- Graph centrality information.

**New Patient Readmission Detection**
- Enter patient details through an interactive form.
- Predict whether the patient is likely to be readmitted.
- Display the estimated readmission probability.

The dashboard can be accessed through a temporary public URL using pyngrok while the Google Colab runtime remains active.

## Results

The project demonstrates:

- Analysis of 101,766 hospital encounters.
- Construction of a patient–diagnosis graph containing 5,542 nodes and 14,650 edges.
- Identification of important diagnosis codes using graph analytics.
- Comparison of Logistic Regression and Random Forest models.
- Achievement of a best ROC-AUC score of 0.6509 using Logistic Regression.
- Development of an interactive Streamlit dashboard for healthcare analytics and readmission prediction.

## Repository Structure

- `data/` – Dataset information and download instructions.
- `notebooks/` – Google Colab notebook containing data preprocessing, graph analytics, and machine learning.
- `app.py` – Streamlit web application.
- `project_outputs/` – Generated charts, model files, and dashboard data.
- `presentation/` – Project presentation.
- `requirements.txt` – Python dependencies.
- `README.md` – Project documentation.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-analytics-readmission-risk.git
cd healthcare-analytics-readmission-risk
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

- Open the notebook in Google Colab or Jupyter Notebook.
- Execute the cells in sequential order.
- The dataset will be fetched automatically from UCI.
- The notebook generates the graph visualizations, machine learning results, trained model, and dashboard data.

### 4. Run the Streamlit Dashboard

After generating the required files in `project_outputs/`, run:

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit to access the dashboard.

For Google Colab access, configure your own ngrok authentication token in the notebook and execute the Streamlit + ngrok setup cell.

## Important Notes

- This project is developed for academic and educational purposes.
- The dashboard uses the saved Random Forest model for new patient predictions.
- The model predicts readmission based on the selected patient features.
- Graph analytics and machine learning are implemented as separate analytical stages in the current version.
- The project is not intended for real-world clinical diagnosis or medical decision-making.
- Never upload private authentication tokens or credentials to GitHub.

## Future Enhancements

- Integrate graph-derived features into the machine learning prediction model.
- Include diagnosis codes and medication-related features in prediction.
- Improve model performance through feature engineering and hyperparameter tuning.
- Deploy the dashboard on a permanent cloud hosting platform.
- Extend graph analytics to explore more complex patient relationships.

## Team Members

1. A Sai Sirish
2. D Manideep
3. K Vignesh Reddy

## Academic Information

**Subject:** Graph and Web Analytics

**Project:** Web-Based Healthcare Analytics and Patient Readmission Risk Prediction

**Platform:** Python, Google Colab, Streamlit
