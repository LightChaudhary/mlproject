# Student Performance Prediction

An end-to-end machine learning application that predicts a student's mathematics score using demographic information, reading score, and writing score.

**Live Demo:** [Student Performance Prediction](https://mlproject-deployment.onrender.com/)

**Prediction Form:** [Open Prediction Form](https://mlproject-deployment.onrender.com/predictdata)

## Overview

This project demonstrates an end-to-end machine learning workflow, including data ingestion, preprocessing, model training, model serialization, and web-based inference.

The application uses a Flask web interface to collect student information and generate a predicted mathematics score using a trained regression model.

## Features

- Data ingestion and train/test splitting
- Numerical preprocessing with median imputation and standardization
- Categorical preprocessing with most-frequent imputation and one-hot encoding
- Comparison of multiple regression algorithms
- Hyperparameter tuning using `GridSearchCV`
- Model and preprocessing artifact serialization
- Flask-based web interface for interactive predictions

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn, CatBoost, XGBoost
- **Data Processing:** pandas, NumPy
- **Web Framework:** Flask
- **Model Serialization:** dill
- **Deployment:** Render

## Project Structure

```text
mlproject/
├── artifacts/              # Saved datasets, model, and preprocessor
├── notebook/               # Exploratory data analysis and experiments
├── src/
│   ├── components/         # Data ingestion, transformation, model training
│   ├── pipeline/           # Training and prediction pipelines
│   ├── exception.py        # Exception handling
│   ├── logger.py           # Logging configuration
│   └── utils.py             # Utility functions
├── templates/              # HTML templates
├── app.py                  # Flask application entry point
├── application.py          # Flask application entry point
├── requirements.txt        # Python dependencies
└── setup.py                # Package configuration
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/LightChaudhary/mlproject.git
cd mlproject
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start the application

```bash
python application.py
```

Open the following URL in your browser:

`http://127.0.0.1:5001/`

To access the prediction form, visit:

`http://127.0.0.1:5001/predictdata`

## Model Workflow

1. Load the student performance dataset.
2. Split the data into training and test sets.
3. Identify numerical and categorical features.
4. Fit preprocessing transformations on the training data.
5. Train and compare candidate regression models using cross-validation and hyperparameter tuning.
6. Save the selected model and fitted preprocessor.
7. Load the saved artifacts to generate predictions from user-provided input.

## Limitations

This is an educational machine learning project. Predictions are estimates and are not guaranteed to reflect a student's actual performance. Model quality depends on the training data and evaluation methodology. The application should not be used as the sole basis for consequential decisions about students.

## Future Improvements

- Improve model selection and final test-set evaluation.
- Add automated tests for preprocessing and prediction.
- Strengthen input validation and error handling.
- Add model performance reporting and experiment tracking.
- Improve reproducibility and deployment documentation.
