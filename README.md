# 🔥 Weather Wildfire Avoidance

**Goal**: Predict dry weather conditions to aid in wildfire prevention efforts using machine learning.

This project is part of my learning journey through the machine learning lifecycle. It focuses on predicting dryness trends that indicate wildfire risk, using weather data inspired by the Canadian Forest Fire Weather Index (FWI) system. It's a hands-on implementation of concepts learned during my Data Science course at CICCC.

---

## 📊 Problem Statement

Unpredicted dry spells increase wildfire risk significantly. This project aims to identify such conditions early using meteorological data. By forecasting "dry" conditions, we can support better decision-making in wildfire resource allocation and early warning systems.

---

## 🌐 Data Overview

The dataset contains historical weather observations with features such as:

- Temperature  
- Humidity  
- Precipitation (%)  
- Wind Speed  
- UV Index  
- Cloud Cover  
- Visibility  
- Season  
- Location  

A label `Dryness_Label_Wildfire` is defined using thresholds based on:
- Temperature > 25°C
- Humidity < 50%
- Precipitation < 10%

---

## 🔍 ML Lifecycle Covered

- ✅ Problem definition  
- ✅ Data loading & cleaning  
- ✅ Feature engineering  
- ✅ Outlier handling  
- ✅ EDA (statistical tests, visualizations)  
- ✅ Label creation  
- ✅ Data splitting  
- ✅ Model evaluation planning  
- 🔄 Model training (starting with multiple linear regression)  
- 🛠️ Deployment strategy planned  

---

## 🧠 Model Training

Currently experimenting with:

- **Multiple Linear Regression**  
Planned future models may include:
- Logistic Regression  
- Decision Trees  
- Random Forest  
- Gradient Boosting  
- SVM  

---

## 📈 Evaluation Metrics

To assess model performance:
- **Recall** (priority to catch dry conditions)  
- **F1 Score**  
- **ROC AUC**  
- Confusion Matrix & Classification Report  

---

## 🚀 Deployment (Planned)

Once the model is validated, the goal is to expose it via a simple API (e.g., Flask or FastAPI), allowing real-time predictions for dryness conditions.

---

## 📚 Tech Stack

- Python (pandas, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook
- Visualizations with Plotly / Seaborn
- Version control with Git

---

## 📌 Status

> Still in progress. Currently exploring model training and tuning.

---

## 🧑‍🎓 Author

**Amir Lima Oliveira**  
Cornerstone International Community College of Canada  
Instructor: Austin Eghbal

---

## 💡 Inspiration

Built using the Canadian Fire Weather Index (FWI) system as a reference for defining fire-risk conditions, connecting environmental science with practical machine learning applications.

