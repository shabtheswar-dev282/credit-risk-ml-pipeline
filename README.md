# 🏦 End-to-End Credit Risk & Loan Allocation Predictive System

![Python Version](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This repository contains a dual-stage machine learning pipeline developed to automate banking loan decisions. The project is divided into two primary case studies:
* **Case Study A (Classification):** Predicting loan approval status by identifying high-risk applicants, with a heavy focus on minimizing false positives (approving bad loans).
* **Case Study B (Regression):** Calculating a safe, personalized maximum loan limit for approved clients, ensuring "white-box" transparency for regulatory financial audits.

**Author:** Deny Shabtheswar  
**Group:** CS-26  

---

## 🛠️ Tech Stack
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Algorithms:** Logistic Regression, Gaussian Naïve Bayes, K-Nearest Neighbours (KNN), Decision Tree Regressor, Soft Voting Ensemble
* **Techniques:** GridSearchCV, Stratified Train-Test Split, StandardScaler, One-Hot Encoding, 99th Percentile Winsorization (Capping)

---

## 📁 Repository Structure

The project is structured into three sequential Jupyter Notebooks to maintain a clean, modular data mining pipeline (following CRISP-DM principles).

### 📓 Notebook 1: Data Understanding & Preprocessing
Handles the initial ~58,000 record dataset, focusing on structural integrity and statistical cleaning.
* **Leakage Prevention:** Removed variables determined *after* loan approval (e.g., `Max Allowed Loan`) to prevent target leakage.
* **Imputation:** Applied Median imputation for missing numeric values (to maintain robustness against massive outliers) and Mode imputation for categorical values.
* **Anomaly Capping:** Removed negative financial values and capped extreme, impossible outliers (e.g., age capped to 75, employment length to 40, and 99th percentile Winsorization for extreme loan amounts).
* **Output:** Generated two pristine datasets: `df_class_clean.csv` and `df_reg_clean.csv`.

### 📓 Notebook 2: Data Cleaning, Modelling, and Evaluation
Focuses on predicting Loan Approval Status dealing with a severe 85:15 class imbalance (85% rejected).
* **Preprocessing:** Applied One-Hot Encoding for categorical features and `StandardScaler` for distance-based algorithms.
* **Data Splitting:** Utilized an 80/20 train-test split with strict **Stratification** to perfectly preserve the 85:15 imbalance in both sets.
* **Base Modelling:** Evaluated Logistic Regression, Gaussian Naïve Bayes, and K-Nearest Neighbours (KNN).
* **Metric Selection:** Specifically optimized for **Recall** and **Precision** (avoiding misleading Accuracy scores), utilizing Confusion Matrices and AUC-ROC curves to select KNN as the best base algorithm.
* **Hyperparameter Tuning:** Applied `GridSearchCV` to maximize Recall for the 'Reject' class, catching high-risk clients aggressively.

### 📓 Notebook 3: Ensemble Classification & Decision Tree Regression
Integrates peer-review feedback and finalizes the predictive outputs.
* **Soft Voting Ensemble:** Combined the linear strengths of Logistic Regression with the distance-based strengths of KNN into a probability-based ensemble, achieving an overall **AUC-ROC of 0.91** and a 'Reject' **Precision of 0.86**.
* **Decision Tree Regression:** Built a fully grown Decision Tree (DT-1) and a pruned tree (DT-2, max depth 3-4) to predict maximum loan limits for approved clients. 
* **Evaluation:** DT-1 was selected as the superior model due to its significantly lower Mean Squared Error (MSE), proving better equipped to map sharp, non-linear financial thresholds.
* **Explainability / Inference:** Predicted the loan limit for Target Client 60256 ($83,097.00) and extracted the exact mathematical decision path (evaluating Income, Age, and Loan-to-Income brackets) to provide "White-Box" transparency.

---

## 📊 Key Findings & Challenges Addressed
1. **Severe Class Imbalance:** Accuracy was discarded as a metric. Optimization heavily favored Recall and Precision to align with the bank's goal of minimizing financial risk.
2. **Extreme Outliers:** Distance-based algorithms (like KNN) and regression metrics (MSE) were protected from distortion through aggressive median imputation and percentile capping.
3. **Model Explainability:** By extracting the decision path from the final Decision Tree Regressor, the system fulfills the requirement for auditable and justifiable financial decisions.

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/Credit-Risk-Prediction-System.git](https://github.com/yourusername/Credit-Risk-Prediction-System.git)

   

1. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn jupyter

2. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Run the notebooks in chronological order:

* Execute Notebook 1 first to generate the cleaned .csv files.

* Execute Notebook 2 for classification analysis.

* Execute Notebook 3 for ensemble evaluation and regression predictions.
   








   
