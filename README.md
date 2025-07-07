![image](https://github.com/user-attachments/assets/2204bb58-c11a-42b9-90b0-3f4870b8faf7)
#  Insurance Cost Tier Classification

## Summary

This repository applies supervised machine learning techniques to classify individuals into insurance cost tiers (low, medium, high) using the Medical Cost Personal Dataset from Kaggle: [View Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

---

##  Overview

###  Challenge Definition

The task is to predict which tier of insurance cost a person will fall into — Low, Medium, or High — based on demographic and lifestyle features. This transforms the regression-based `charges` prediction into a **multiclass classification** task for better interpretability and segmentation.

### Approach

We reframed the dataset using cost tier binning and trained models including **Logistic Regression**, **Random Forest**, and **K-Nearest Neighbors (KNN)**. Each model was trained using pipelines with preprocessing, followed by **GridSearchCV-based hyperparameter tuning**.

###  Performance Summary

**Best Model: Tuned Random Forest**
- **Validation Accuracy:** 91.04%
- **Macro F1 Score:** 0.91

This model outperformed all others across accuracy, precision, and F1 score metrics.

---

##  Summary of Work Done

###  Data

- **Type:** Tabular CSV (demographic and health data)
- **Input:** Age, sex, smoker status, region, BMI, children
- **Output:** Insurance cost tier (`Low`, `Medium`, `High`)
- **Size:** 1,338 records
- **Split:** 802 train / 268 validation / 268 test (stratified)
- 
### Preprocessing / Cleanup
* One-hot encoding of categorical features
* Standard scaling of numerical features
* Quantile binning of `charges` into three cost tiers
* Winsorization to control outliers in BMI

### Data Visualization
* **Histograms:** Distribution of age, BMI, and children by cost tier
* **Count plots:** Smoker, region, and sex distributions by cost tier

![image](https://github.com/user-attachments/assets/4924f023-0a0d-4a24-bc98-ece820df0e0a)
![image](https://github.com/user-attachments/assets/cceda7f0-b064-4ce4-971f-f58a15363cdc)
![image](https://github.com/user-attachments/assets/d912b4a1-1a64-424d-aee9-aec7e9b88645)
![image](https://github.com/user-attachments/assets/533048f0-9b34-4300-96e4-5549217e06c0)
![image](https://github.com/user-attachments/assets/f18ec0d1-7834-47de-a0ad-538860d9531e)
![image](https://github.com/user-attachments/assets/b79f6e10-a4e6-4913-9794-4afe867d0d07)
![image](https://github.com/user-attachments/assets/aaa47fe1-8d63-4672-afe8-cf3bb95acf78)
### Numerical Features – Histogram Distributions by Cost Tier
 visualized all numerical features using grouped histograms overlaid by `cost_tier`. Each subplot shows how the values of a given feature are distributed across the three insurance cost tiers (`Low`, `Medium`, `High`).

**Observations:**
- **`bmi`** shows a noticeable rightward shift for the High tier. Individuals in this group tend to have higher BMI values, which may correlate with higher medical risk.
- **`age`** also trends older in the High tier, suggesting age plays a significant role in determining insurance cost.
- **`children`** has a more balanced distribution and shows a weaker relationship with the target, but the Medium tier has a wider spread.

This helps identify which features may carry more weight in prediction.
###  Categorical Features – Count Plots by Cost Tier

visualized the categorical features using count plots, grouped by `cost_tier` using `hue`.

**Observations:**
- **`smoker`** stands out as the most predictive categorical feature. Nearly all smokers fall into the High cost tier, showing strong influence.
- **`sex`** and **`region`** show relatively balanced distributions across cost tiers, suggesting they are likely less influential features on their own.
- These plots confirm the importance of visual EDA in identifying which features are likely to improve model performance.

### Visualization Summary

- Histogram and bar plots revealed key relationships between features and the target.
- **Strong signals:** `smoker`, `bmi`, and `age`.
- **Weaker signals:** `region`, `children`, and possibly `sex`.


##  Problem Formulation

- **Input:** Demographic and lifestyle features
- **Output:** Multiclass label (`Low`, `Medium`, `High`)
- **Task:** Supervised classification

---

##  Models

###  Models Trained

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)

<img width="558" alt="image" src="https://github.com/user-attachments/assets/b3e4642f-ab53-4aca-afb4-c3b69f721f5e" />

### Logistic Regression – Performance Summary

- The Logistic Regression model achieved **83.96% validation accuracy** and a **macro F1 score of 0.83**, which is excellent for a 3-class classification task.
- **Low and Medium tiers** were predicted very well:
  - `Low`: F1 = 0.90
  - `Medium`: F1 = 0.84, with extremely high recall (0.97)
- **High tier** had the highest precision (0.98) but lower recall (0.62), meaning the model is cautious in assigning this label.
- These results suggest the model effectively captures cost patterns based on features like `smoker`, `bmi`, and `age`.
- The confusion matrix reveals that **most errors happen with High tier**, which may require additional feature engineering or model tuning in future iterations.

  <img width="485" alt="image" src="https://github.com/user-attachments/assets/65dfa0ec-b46b-4e8b-9d15-4052aae45c5a" />
  ###  Random Forest – Performance Summary

The **Random Forest** model achieved **90% validation accuracy** and a **macro F1 score of 0.90**, making it the strongest performer among all models tested.

- **High Tier:** F1 = **0.91**, with high precision (0.94) and recall (0.89)
- **Low Tier:** F1 = **0.90**, with strong recall (0.93)
- **Medium Tier:** F1 = **0.89**, with balanced precision and recall

Random Forest performs exceptionally well due to its ensemble learning approach, which enhances stability and generalization across both numerical and categorical features.

The confusion matrix reveals that most misclassifications occur between **Medium and High tiers**, which is expected due to feature overlap.

This model is a strong candidate for deployment given its **robust performance, interpretability**, and **high accuracy across all classes**.
<img width="449" alt="image" src="https://github.com/user-attachments/assets/41692332-08ac-4c87-805b-92841aec97f2" />
### K-Nearest Neighbors (KNN) – Performance Summary

The **KNN** model achieved **82% validation accuracy** and a **macro F1 score of 0.82**, performing comparably to Logistic Regression.

- **High Tier:** F1 = **0.80**, with moderate recall (0.74), showing it occasionally confuses High with Medium or Low.
- **Low Tier:** F1 = **0.85**, with strong recall (0.88), making it the most confidently predicted class.
- **Medium Tier:** F1 = **0.80**, with decent performance, though some misclassification still occurs.

KNN is easy to implement and interpret but can be sensitive to feature scaling and class overlap.  
The confusion matrix suggests **Medium and High tiers** are frequently confused — likely due to shared characteristics in features like `age`, `BMI`, or `smoker`.

 While KNN may not be the top performer, it provides a **solid, interpretable benchmark** and can still contribute to ensemble modeling or hybrid pipeline

 ###  Model Comparison – Summary (Non-Tuned)

This section compares the performance of three baseline models: **Logistic Regression**, **Random Forest**, and **K-Nearest Neighbors (KNN)** on the task of classifying insurance cost tiers.

| Model                | Accuracy | Macro F1 Score |
|---------------------|----------|----------------|
| Logistic Regression | 83.96%   | 0.83           |
| Random Forest       | 90.00%   | 0.90           |
| KNN                 | 82.00%   | 0.82           |

![download](https://github.com/user-attachments/assets/2c4cd3d6-2bce-46ca-88ce-6af225e5de52)


####  Key Observations:
- **Random Forest** is the best-performing model overall with the highest accuracy and F1 score, showing strong generalization across all tiers.
- **Logistic Regression** performs well, especially on the Low and Medium tiers, with a very high precision for High tier but slightly lower recall.
- **KNN**, while easy to implement and interpret, shows slightly lower performance, struggling more with class overlap—especially between Medium and other tiers.

These results provide a strong starting point. Further performance gains may be achieved through hyperparameter tuning, feature engineering, or ensemble methods. 



###  Hyperparameters Tuned

- **Logistic Regression:** `C`, `penalty`, `solver`
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
- **KNN:** `n_neighbors`, `weights`, `metric`

---

## Training
* **Software:** Python 3, scikit-learn, pandas, matplotlib, seaborn
* **Environment:** Jupyter Notebook (Anaconda/Kaggle)
* **Training Method:** 5-fold GridSearchCV for hyperparameter optimization
* **Duration:** Each model trained in under 10 minutes
---

##  Performance Comparison

| Model                  | Accuracy | Macro F1 Score |
|------------------------|----------|----------------|
| Logistic Regression    | 0.87     | 0.87           |
| Random Forest (Tuned)  | **0.91** | **0.91**       |
| KNN (Tuned)            | 0.85     | 0.85           |

![download](https://github.com/user-attachments/assets/b269f2b1-43c8-4c43-9a4a-80d0f2a4f983)


 Precision for the **High tier** was exceptionally high (**1.00**), but its recall was more modest (**0.72**), indicating the model is cautious in labeling someone as "High" cost.

This tuned model leveraged **L2 regularization** and **lower regularization strength (C=0.1)** to generalize better than the untuned version.

![download](https://github.com/user-attachments/assets/3c067d96-2efc-4602-b5de-745367a8e091)


The model was tuned with:
- `n_estimators = 200`
- `max_features = 'sqrt'`
- `min_samples_split = 5`

It shows excellent class balance and minimal misclassification, especially between overlapping tiers like Medium and High. 

![download](https://github.com/user-attachments/assets/68c52f9f-816c-495a-ac4e-2b2fed8ceea1)


The **tuned KNN** model achieved **84.70% validation accuracy** and a **macro F1 score of 0.85**.

- **High Tier:** F1 = **0.85**
- **Low Tier:** F1 = **0.87**
- **Medium Tier:** F1 = **0.82**

 Best Parameters:
- `n_neighbors = 9`
- `weights = 'distance'`
- `metric = 'euclidean'`

 While performance improved over the baseline, the Medium tier still shows overlap. KNN remains useful for fast and interpretable predictions.


- Random Forest showed consistently strong results across all classes.



---

##  Conclusions

- The **Tuned Random Forest** model offers the best balance between performance and robustness.
- Logistic Regression had high precision but slightly lower recall on the "High" tier.
- KNN was interpretable and lightweight, but sensitive to scaling and neighborhood size.

---

##  Future Work

- Add SHAP or LIME for prediction explanations.
- Try gradient boosting algorithms like XGBoost or LightGBM.
- Integrate additional socioeconomic features like income or location.
- Deploy model using Flask or FastAPI.

## How to Reproduce Results

### Clone Repository
```bash
git clone https://github.com/yourusername/insurance-cost-tier-classification.git
cd insurance-cost-tier-classification
### Step 2 pip install -r requirements.txt
### Step 3: Download Dataset
Download insurance.csv and place it in the project root

Step 4: Run Notebook
Open Cost Tiers.ipynb and run all cells to:

Preprocess data

Train models

Visualize results

View final evaluation
---

.
├── insurance.csv             # Raw dataset from Kaggle
├── Cost Tiers.ipynb          # Main notebook with full project pipeline
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies




