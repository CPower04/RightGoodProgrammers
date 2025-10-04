# RightGoodProgrammers - HackAthlone 2025
## Project: *A World Away: Hunting for Exoplanets with AI*

---

## Team Members
- **Sean Sibindi**
- **Conor Power**
- **Philip Roche**
- **Robert Shanahan**
- **Adam Downes**

---

## 🌌 Project Overview
**Goal:** Build an application that takes CSV datasets of potential exoplanets and uses a pre-trained AI model to classify them, identifying promising candidates for further study.

**Datasets Used:**

| Dataset      | Description                    |
|--------------|--------------------------------|
| KOI          | Kepler Objects of Interest     |
| TOI          | TESS Objects of Interest       |
| K2 Planets   | K2 Planets and Candidates      |

---

## 🧹 Data Cleaning Workflow

### Loading Data
- Imported CSV files into **pandas DataFrames**.

### Column Selection
- Removed irrelevant columns.  
- Dropped columns with a high proportion of null values.

### Handling Null Values
- Identified missing data.  
- Removed columns exceeding a set null threshold.

### Target Encoding
- Converted categorical labels into numerical values using **one-hot encoding**.

### Correlation Analysis
- Used **scikit-learn** to drop highly correlated features.  
- For TOI, `rastr` and `decstr` columns (string format) were converted into **float seconds** for correlation calculations.

✅ **Result:** Cleaned datasets ready for model training.

---

## 🤖 Model Training Pipeline

### Feature & Label Separation
- Identified **X (features)** and **Y (target labels)**.  
- Split into training and testing sets with `train_test_split`.

### Scaling & Normalization
- Ensured no single feature disproportionately influenced the model.

### Label Encoding
- Combined encoded content into a single column for training.

### Model Selection & Training
- Used **RandomForestClassifier** from **scikit-learn**.  
- Applied **hyperparameter tuning** to optimize performance.  
- Trained multiple models per dataset and selected the **best F1-score model**.

### Evaluation
- Generated **confusion matrices** to visualize classification performance.


