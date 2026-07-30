# Impact of AI Tool Usage on Academic Performance

An end-to-end machine learning project utilizing **XGBoost** to analyze and predict the impact of student AI tool adoption (e.g., ChatGPT, coding assistants, study aids) on academic outcomes. This project includes data cleaning, feature engineering, model training, hyperparameter tuning, and model explainability using SHAP values.

---

## 📌 Project Overview

As generative AI tools become ubiquitous in modern education, understanding their concrete effect on student performance is critical. This project models the relationship between various AI usage metrics—such as frequency, prompt complexity, and specific task use cases—and final student grades/GPA. 

Using **XGBoost**, the pipeline achieves high predictive accuracy while maintaining interpretability to identify which specific AI study habits positively or negatively correlate with academic success.

---

## 🛠️ Key Features

* **Data Preprocessing & Feature Engineering:** Cleaned raw survey/log data, handled missing values, encoded categorical variables, and engineered interaction terms representing usage density.
* **Exploratory Data Analysis (EDA):** Visualized distribution trends, correlation matrices, and grade shifts across different academic disciplines.
* **Model Training & Optimization:** Implemented `XGBoostClassifier` alongside baseline models (Linear Regression, Decision Trees, Random Forest). Tuned hyperparameters using `GridSearchCV`.
* **Model Explainability (SHAP):** Leveraged **SHAP (SHapley Additive exPlanations)** to break open the "black box" and rank feature importance for actionable insights.

---

## 💻 Tech Stack

* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`
* **Machine Learning:** `xgboost`, `scikit-learn`
* **Model Interpretability:** `shap`
* **Visualization:** `matplotlib`, `seaborn`

---

## 📂 Project Structure

```text
├── data/
│   ├── raw_student_data.csv        # Raw dataset
│   └── processed_student_data.csv  # Cleaned & feature-engineered dataset
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb# Preprocessing & encoding pipeline
│   └── 03_xgboost_modeling.ipynb   # Training, tuning, & SHAP interpretation
├── models/
│   └── xgboost_best_model.pkl      # Saved trained model artifact
├── src/
│   ├── data_loader.py              # Data loading utility scripts
│   └── evaluate.py                 # Evaluation metrics calculation
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
