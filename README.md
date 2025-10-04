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
```python
threshold = int(np.ceil(len(cumulative_df) * 0.75))  # require ≥75% non-missing to keep
cumulative_df = cumulative_df.dropna(thresh=threshold, axis=1)  
```
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

```python
import numpy as np
import pandas as pd

# Compute the absolute correlation matrix
corr_matrix = X.corr().abs()

# Select the upper triangle of the correlation matrix
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify columns with correlation greater than 0.95
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

# Drop the highly correlated columns
X = X.drop(columns=to_drop)

print("\nDropped columns due to high correlation:", to_drop)
```
- Split into training and testing sets with `train_test_split`.

### Scaling & Normalization
- Ensured no single feature disproportionately influenced the model.

### Label Encoding
- Combined encoded content into a single column for training.

### Model Selection & Training
- Used **RandomForestClassifier** from **scikit-learn**. 
- Applied **hyperparameter tuning** to optimize performance.  
- Trained multiple models per dataset and selected the **best F1-score model**.
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = best_rf.predict(X_test_scaled)

cm = confusion_matrix(y_test_enc, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.savefig(os.path.join(dir_path.replace('data', 'models'), 'confusion_matrix.png'))
```
### Evaluation
- Generated **confusion matrices** to visualize classification performance.
### Readings
- Inspiration and background knowledege was obtained from reading : Humphrey, A.L.T. & Quintana, E.V., 2020. Predicting missing planets in multiplanet system populations via analytical assessments of dynamical packing. arXiv preprint arXiv:2011.03053 [astro-ph.EP]. Available at: https://arxiv.org/abs/2011.03053
 [Accessed 4 October 2025].


