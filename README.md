# Credit Card Approval Prediction

> Automating credit card application decisions using Logistic Regression.

---

## Problem Statement

A major financial institution receives a high volume of credit card applications, and the manual review process is **time-consuming and inconsistent**. The goal is to build an automated system that predicts whether a credit card application should be **approved or rejected** based on applicant information such as income, credit score, debt ratio, and other attributes.

---

## Objective

Build a **Logistic Regression** model to predict `approval_status` (1 = Approved, 0 = Rejected), optimizing for the **Weighted F1 Score**.

---

## Dataset

| File                    | Shape    | Description                          |
|-------------------------|----------|--------------------------------------|
| `train.csv`             | 101 × 14 | Labeled training data                |
| `test.csv`              | 151 × 13 | Unlabeled test data (for prediction) |
| `sample_submission.csv` | 3 × 2    | Expected submission format           |

### Features

| Column                 | Type        | Description                                               |
|------------------------|-------------|-----------------------------------------------------------|
| `application_id`       | ID          | Unique identifier for each application                    |
| `age`                  | Numerical   | Age of the applicant (years)                               |
| `income`               | Numerical   | Annual income (USD)                                        |
| `employment_year`      | Numerical   | Years in current employment                                |
| `debt_to_income_ratio` | Numerical   | Monthly debt payments / monthly income                     |
| `credit_score`         | Numerical   | Credit score (300–850)                                     |
| `loan_amount`          | Numerical   | Loan amount requested (USD)                                |
| `loan_term`            | Numerical   | Loan term (months)                                         |
| `has_mortgage`         | Binary      | Has a mortgage? (1 = Yes, 0 = No)                         |
| `has_dependents`       | Binary      | Has dependents? (1 = Yes, 0 = No)                         |
| `has_prev_default`     | Binary      | Has previous defaults? (1 = Yes, 0 = No)                  |
| `education_level`      | Categorical | Education level of the applicant                           |
| `marital_status`       | Categorical | Marital status of the applicant                            |
| `approval_status`      | Target      | **1** = Approved, **0** = Rejected *(training data only)*  |

---

## Tech Stack

| Library        | Purpose                                   |
|----------------|-------------------------------------------|
| Pandas         | Data loading & manipulation               |
| Scikit-learn   | Label encoding, model training & prediction |

---

## Approach

```
Load train.csv & test.csv
       ↓
Encode Categoricals (LabelEncoder)
   ├── education_level
   └── marital_status
       ↓
Prepare Features
   ├── X = all columns except application_id & approval_status
   └── y = approval_status
       ↓
Train Logistic Regression (max_iter=1000)
       ↓
Predict on test set → submission.csv (151 × 2)
```

### Steps

1. **Data Loading** — Loaded `train.csv` and `test.csv` using Pandas.
2. **Categorical Encoding** — Applied `LabelEncoder` to `education_level` and `marital_status` to convert them into numerical form.
3. **Feature Preparation** — Dropped `application_id` and `approval_status` from training features; dropped `application_id` from test features.
4. **Model Training** — Trained a `LogisticRegression` model with `max_iter=1000` on all training data.
5. **Prediction & Submission** — Generated predictions on the test set and exported `submission.csv` with `application_id` and `approval_status` columns.

---

## Evaluation Metric

**Weighted F1 Score:**

```
Score = max(0, 100 − (MAE(actual, predicted) ^ 0.5) / 10)
```

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install pandas scikit-learn
   ```

2. **Place dataset files** in a `dataset/` folder:
   ```
   dataset/
   ├── train.csv
   ├── test.csv
   └── sample_submission.csv
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook jupyter_notebook.ipynb
   ```

4. **Output** — `submission.csv` (151 × 2) is generated in the root directory.

---

## Project Structure

```
Predict-credit-card-approval/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── jupyter_notebook.ipynb    # Solution notebook
├── submission.csv            # Generated predictions
└── README.md
```

---

## Constraints

| Constraint     | Value    |
|----------------|----------|
| Time Limit     | 5.0 sec  |
| Memory Limit   | 256 MB   |
| Source Limit    | 1024 KB  |

---

## License

MIT License - See LICENSE file for details
