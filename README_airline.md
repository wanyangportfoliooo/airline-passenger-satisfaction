# Airline Passenger Satisfaction – ML Classification

## Project Overview
This project builds a supervised machine learning pipeline to predict airline passenger satisfaction (satisfied vs. neutral/dissatisfied) using a labelled dataset of passenger feedback and flight information.

## Dataset
- **Training set**: `Assignment3-TrainingDataset.csv`
- **Unknown set**: `Assignment3-UnknownDataset.csv`
- **Features**: Passenger demographics, flight details, and 14 service rating scores (Likert scale 1–5)
- **Target**: `satisfaction` (satisfied / neutral or dissatisfied)

## Tech Stack
- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

## Methodology
1. **Exploratory Data Analysis (EDA)** – distribution of target variable, missing value check
2. **Data Preprocessing** – median/mode imputation, label encoding, standardisation
3. **Feature Engineering** – delay aggregation (Total_Delay, Has_Delay, Severe_Delay), service score aggregation (Avg_Service_Score, Total_Service_Score, High/Low_Score_Count)
4. **Model Comparison** – 9 classifiers evaluated using stratified 5-fold cross-validation and 80/20 hold-out split:
   - Decision Tree
   - k-Nearest Neighbors (k=7)
   - Artificial Neural Network (MLP)
   - Linear Discriminant Analysis
   - Perceptron
   - Support Vector Machine (RBF kernel)
   - Random Forest
   - AdaBoost
   - Gradient Boosting
5. **Evaluation Metrics** – ROC-AUC (primary), F1, Accuracy, Precision, Recall
6. **Best Model** – Random Forest (ROC-AUC: 0.9932, F1: 0.9539, Accuracy: 96.07%)
7. **Threshold Optimisation** – adjusted from 0.50 → 0.49 using Youden's J and F1 maximisation to reduce false negatives

## Results
| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| Random Forest | 0.9932 | 0.9539 | 96.07% |
| ANN (MLP) | 0.9921 | 0.9454 | 95.31% |
| Gradient Boosting | 0.9885 | 0.9334 | 94.30% |

## How to Run
1. Clone this repository
2. Place `Assignment3-TrainingDataset.csv` and `Assignment3-UnknownDataset.csv` in the same folder as the script
3. Install dependencies:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```
4. Run the script:
```
python airline_satisfaction_fixed.py
```
