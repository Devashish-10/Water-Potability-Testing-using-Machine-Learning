
# Water Potability Prediction Project

This project aims to build a machine learning model to predict the potability of water using multiple algorithms, including **Support Vector Machine (SVM)**, **Random Forest**, and **XGBoost**. The dataset used in this project contains information about various water quality parameters and a target variable indicating whether the water is potable or not.

## Table of Contents
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Workflow](#project-workflow)
- [Usage](#usage)
- [Results](#results)
- [Model Storage](#model-storage)

## Dataset
The dataset used in this project is named `water_potability.csv`. It contains the following features:
- Various water quality parameters (e.g., pH, Hardness, Solids, Chloramines, etc.).
- A target variable: `Potability` (0 for non-potable water, 1 for potable water).

## Dependencies
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `imblearn`
- `xgboost`
- `shap`
- `joblib`
- `pickle`

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap joblib
```

## Project Workflow
1. **Loading the Dataset**:
   - The dataset is loaded using `pandas`.
   - Display the first few rows to understand the structure of the data.

2. **Handling Missing Values**:
   - Missing values are imputed using the mean strategy.

3. **Feature Scaling**:
   - The features are scaled using `StandardScaler` to ensure uniformity across the dataset.

4. **Handling Class Imbalance**:
   - The Synthetic Minority Over-sampling Technique (SMOTE) is used to handle class imbalance.

5. **Data Splitting**:
   - The preprocessed data is split into training and testing sets (80:20 split).

6. **Model Training**:
   - Multiple models are trained, including:
     - Support Vector Machine (SVM)
     - Random Forest
     - XGBoost
   - An ensemble model using `VotingClassifier` is also created.

7. **Model Evaluation**:
   - The models' accuracy is evaluated using the `accuracy_score` metric, along with confusion matrices and classification reports.

8. **Model Saving**:
   - The best ensemble model is saved as `waterpotability_voting_model.pkl` using the `pickle` library.

## Usage
If you are looking to use this Program
1. Clone this repository and navigate to the project directory:
   ```bash
   git clone <repository-link>
   cd <project-directory>
   ```

2. Place the `water_potability.csv` file in the project directory.

3. Run the script:
   ```bash
   python water_potability_prediction.py
   ```

4. The script will:
   - Preprocess the dataset.
   - Train multiple models.
   - Display the accuracy of each model.
   - Save the best ensemble model as `waterpotability_voting_model.pkl`.

## Results
The ensemble model achieves an accuracy of **73.38%%** (or the actual accuracy based on your latest results).

## Model Storage
The trained ensemble model is saved as `waterpotability_voting_model.pkl`. It can be loaded in future scripts for making predictions without retraining the model:
```python
import pickle
model = pickle.load(open('waterpotability_voting_model.pkl', 'rb'))
```

## Note
This project was created as a practice exercise to enhance machine learning skills and is not intended for production use or real-world applications.
