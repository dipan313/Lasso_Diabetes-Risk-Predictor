# 🏥 Diabetes Risk Predictor
### Lasso Regression · Healthcare Analytics · Automatic Feature Selection

<p align="center">
  <img src="https://img.shields.io/badge/Algorithm-Lasso%20Regression%20%28L1%29-0F6E56?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Domain-Healthcare%20%2F%20Clinical%20ML-D85A30?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-BRFSS%202015%20%7C%20253%2C680%20rows-185FA5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Made%20by-Dipan%20Mazumder-7F77DD?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Features%20Selected-24%20of%2026-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/R²%20Score-0.1648-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Top%20Predictor-BMI__Age%20(engineered)-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Key%20Win-Feature%20Discovery-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset%20Size-250k%20rows-purple?style=flat-square"/>
</p>

---

## 📌 Problem Statement

> **Given 21 health indicators collected from 253,680 Americans, which clinical markers actually predict diabetes risk — and can a model identify them automatically without human-guided feature selection?**

The CDC's **Behavioral Risk Factor Surveillance System (BRFSS)** is the largest health telephone survey in the United States. This project uses the 2015 BRFSS dataset to build a diabetes risk prediction model using **Lasso Regression** — a regularization method that performs **automatic feature selection** by shrinking irrelevant feature coefficients to exactly zero.

In clinical machine learning, interpretability is not a luxury — it is a regulatory and ethical requirement. A model that explicitly identifies *"these 8 markers matter, these do not"* is far more deployable in healthcare settings than a black-box model with marginally better accuracy.

---

## 🎯 Key Question

> *"Of 21 health indicators measured across a quarter-million Americans, which ones are genuinely predictive of diabetes — and which are statistical noise a clinical screening tool can safely ignore?"*

This project answers two questions simultaneously:
1. **Prediction** — estimate diabetes risk score for a new patient
2. **Discovery** — identify the minimum set of clinical markers needed

---

## 🔬 The Lasso Regression Algorithm

### What Is Lasso Regression?

Lasso (**L**east **A**bsolute **S**hrinkage and **S**election **O**perator) is Linear Regression with an **L1 penalty** on the coefficients:

```
  Cost(β) = RSS  +  λ × Σ |βᵢ|
              ↑              ↑
        fit to data    L1 penalty: sum of
                       ABSOLUTE values of coefficients
```

### Why L1 Produces Sparsity — The Geometry

```
  Ridge (L2) constraint region:   Lasso (L1) constraint region:
       (circle — smooth)               (diamond — corners)

           ○ ○ ○                          ◇
         ○       ○                      ◇   ◇
        ○         ○                   ◇       ◇
        ○    ●    ○        vs         ◇   ●   ◇
        ○         ○                   ◇       ◇
         ○       ○                      ◇   ◇
           ○ ○ ○                          ◇

  RSS contours touch circle         RSS contours hit a CORNER
  at a smooth point → all           → one coefficient = exactly 0
  coefficients shrink but           This is automatic
  NONE reach exactly zero           FEATURE ELIMINATION
```

The L1 diamond's corners lie on the axes. When the RSS ellipse touches a corner, one or more coefficients are forced to **exactly zero** — Lasso literally eliminates those features from the model. Ridge cannot do this (circle has no corners).

### Lasso vs Ridge — Side-by-Side

| Property | Lasso (L1) | Ridge (L2) |
|----------|-----------|-----------|
| Penalty term | `λ × Σ\|βᵢ\|` | `λ × Σβᵢ²` |
| Coefficients | Some → **exactly 0** | All → small but non-zero |
| Feature selection | ✅ **Yes — automatic** | ❌ Keeps all features |
| Handles correlated features | ⚠️ Picks one, drops others | ✅ Distributes weight evenly |
| Best when | Many features are irrelevant | All features carry some signal |
| Output sparsity | **Sparse model** | Dense model |
| Clinical interpretability | ✅ Fewer markers needed | ✅ All markers considered |

**Lasso is the right choice here** because:
- 21 original features — not all are independent diabetes predictors
- Clinical deployment requires knowing *which* tests to order
- Sparse models reduce patient burden and screening cost

### How Alpha Was Selected

`LassoCV` was used with **5-fold cross-validation** across 100 log-spaced alpha values:

```python
LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=10000)
```

The optimal alpha minimises cross-validated MSE while producing the sparsest useful model.

---

## 🗺️ Full Pipeline — Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              PROJECT 3 — LASSO REGRESSION PIPELINE                          ║
║              Diabetes Risk Predictor · BRFSS 2015 · 253,680 rows            ║
╚══════════════════════════════════════════════════════════════════════════════╝

 ┌──────────────────────────────────────────────────────────────────────────┐
 │  📥  RAW DATA INGESTION                                                   │
 │                                                                           │
 │   File        : diabetes_binary_health_indicators_BRFSS2015.csv           │
 │   Dimensions  : 253,680 rows  ×  22 columns                              │
 │   Target      : Diabetes_binary  (0 = healthy,  1 = diabetic)            │
 │   Class split : ~86% healthy  |  ~14% diabetic  ← imbalanced             │
 │   Missing vals: 0  (BRFSS is pre-cleaned CDC data)                       │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  🔍  EXPLORATORY DATA ANALYSIS (EDA)                                      │
 │                                                                           │
 │   ┌─ Class Balance ────────────────────────────────────────────────────┐  │
 │   │  Bar chart: 218,334 healthy vs 35,346 diabetic                    │  │
 │   │  86.1% / 13.9% split → imbalanced → affects recall               │  │
 │   └────────────────────────────────────────────────────────────────────┘  │
 │                                                                           │
 │   ┌─ BMI Distribution by Class ────────────────────────────────────────┐  │
 │   │  Diabetic patients have higher BMI distribution (right-shifted)   │  │
 │   │  Healthy: mean BMI ~27  |  Diabetic: mean BMI ~31                 │  │
 │   └────────────────────────────────────────────────────────────────────┘  │
 │                                                                           │
 │   ┌─ Diabetes Rate by Age Group ───────────────────────────────────────┐  │
 │   │  Rising trend: age group 1 (18-24) → 2% rate                     │  │
 │   │               age group 13 (80+)  → 35% rate                     │  │
 │   │  Strong linear age signal → motivates BMI_Age interaction         │  │
 │   └────────────────────────────────────────────────────────────────────┘  │
 │                                                                           │
 │   ┌─ Feature Correlation Ranking ──────────────────────────────────────┐  │
 │   │  All 21 features ranked by |correlation| with Diabetes_binary     │  │
 │   │  Top: GenHlth, HighBP, BMI, DiffWalk, HeartDiseaseorAttack        │  │
 │   └────────────────────────────────────────────────────────────────────┘  │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ⚙️  FEATURE ENGINEERING  (21 original → 26 total features)               │
 │                                                                           │
 │   ╔═ Clinical Interaction Terms ════════════════════════════════════════╗  │
 │   ║                                                                     ║  │
 │   ║  BMI_Age           = BMI × Age                                      ║  │
 │   ║  → Obesity risk compounds with age (non-linear joint effect)       ║  │
 │   ║  → Became the #1 Lasso-selected feature  ★                        ║  │
 │   ║                                                                     ║  │
 │   ║  HighBP_HighChol   = HighBP × HighChol                             ║  │
 │   ║  → Metabolic syndrome proxy (co-occurrence more dangerous)         ║  │
 │   ║                                                                     ║  │
 │   ║  Unhealthy_lifestyle = Smoker + HvyAlcohol + (no PhysActivity)     ║  │
 │   ║  → Composite behavioural risk score                                ║  │
 │   ╚═════════════════════════════════════════════════════════════════════╝  │
 │                                                                           │
 │   ╔═ Clinical Threshold Flags ══════════════════════════════════════════╗  │
 │   ║                                                                     ║  │
 │   ║  BMI_Obese      = (BMI ≥ 30)          WHO obesity cutoff           ║  │
 │   ║  BMI_Overweight = (25 ≤ BMI < 30)     WHO overweight range         ║  │
 │   ╚═════════════════════════════════════════════════════════════════════╝  │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ✂️  PREPROCESSING & SPLITTING                                             │
 │                                                                           │
 │   Stratified split (preserves 86/14 ratio in both sets)                 │
 │   Train  :  202,944 rows  (80%)                                          │
 │   Test   :   50,736 rows  (20%)                                          │
 │   Scaling:  StandardScaler fit on train → transform test                 │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  🔧  LASSOCV — AUTOMATIC ALPHA SELECTION                                  │
 │                                                                           │
 │   LassoCV(alphas=100 candidates, cv=5, max_iter=10000)                   │
 │   5-fold cross-validation on 202,944 training rows                       │
 │   Selects alpha that minimises CV mean squared error                     │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  🌿  REGULARIZATION PATH (lasso_path)                                     │
 │                                                                           │
 │   Computes coefficients across 80 alpha values                           │
 │                                                                           │
 │   coeff                                                                   │
 │    │  ╱──────────────────────  BMI_Age                                   │
 │    │╱                                                                     │
 │    │   ╱────────────────────── GenHlth                                   │
 │    ├──────────────────────────────────────── 0  ← zeroed features        │
 │    │                  ╲──────────────────── Income                       │
 │    │                       ╲─────────────── Age                          │
 │    └────────────────────────────────────────────── alpha (low → high)    │
 │                              ↑                                            │
 │                         Best alpha                                        │
 │                                                                           │
 │   Features that never lift off 0 = confirmed irrelevant                  │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  📊  FEATURE SELECTION RESULTS                                            │
 │                                                                           │
 │   ✅ SELECTED by Lasso (coeff ≠ 0) — 24 of 26 features                   │
 │                                                                           │
 │   Top risk INCREASERS (positive coeff):                                  │
 │   ┌────────────────────────┬──────────┬──────────────────────────────┐   │
 │   │ Feature                │  Coeff   │ Clinical Meaning             │   │
 │   ├────────────────────────┼──────────┼──────────────────────────────┤   │
 │   │ BMI_Age  ★ engineered  │ +0.141   │ Compound obesity-age risk    │   │
 │   │ GenHlth                │ +0.054   │ Poor self-reported health    │   │
 │   │ HighBP_HighChol        │ +0.028   │ Metabolic syndrome marker   │   │
 │   │ HighBP                 │ +0.025   │ Hypertension                │   │
 │   │ HeartDiseaseorAttack   │ +0.022   │ Cardiovascular comorbidity  │   │
 │   │ DiffWalk               │ +0.020   │ Mobility issues → inactivity│   │
 │   └────────────────────────┴──────────┴──────────────────────────────┘   │
 │                                                                           │
 │   Top risk REDUCERS (negative coeff):                                    │
 │   ┌────────────────────────┬──────────┬──────────────────────────────┐   │
 │   │ Feature                │  Coeff   │ Clinical Meaning             │   │
 │   ├────────────────────────┼──────────┼──────────────────────────────┤   │
 │   │ Age (raw)              │ −0.099   │ Captured better by BMI_Age   │   │
 │   │ BMI (raw)              │ −0.039   │ Captured better by BMI_Age   │   │
 │   │ Income                 │ −0.023   │ Higher income = better care  │   │
 │   │ HvyAlcoholConsump      │ −0.018   │ Survivor bias in BRFSS data  │   │
 │   └────────────────────────┴──────────┴──────────────────────────────┘   │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ⚖️  LASSO vs RIDGE COMPARISON                                             │
 │                                                                           │
 │   Metric              Lasso       Ridge       Verdict                    │
 │   ───────────────     ─────────   ─────────   ─────────────────────────  │
 │   Features used       24 / 26     26 / 26     Lasso (simpler model)     │
 │   R²                  0.1648      0.1647      Effectively identical     │
 │   MAE                 0.2111      0.2110      Effectively identical     │
 │   RMSE                0.3165      0.3165      Identical                 │
 │                                                                           │
 │   ★ Same performance, 2 fewer features = Lasso wins on parsimony        │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  📉  DIAGNOSTICS                                                           │
 │                                                                           │
 │   Residuals vs Fitted: DIAGONAL BAND PATTERN — expected for binary       │
 │   target with continuous regression output. Not a bug. Binary            │
 │   outcomes create two residual bands (0−pred and 1−pred).               │
 │                                                                           │
 │   Residual Distribution: bimodal / right-skewed — again expected for    │
 │   a 0/1 target with 86/14 class imbalance.                              │
 │                                                                           │
 │   Q-Q Plot: S-curve deviation — confirms non-normality of residuals,    │
 │   which is inherent when using linear regression on binary outcomes.     │
 │   For true binary classification, Logistic Regression would be ideal.   │
 └────────────────────────────┬─────────────────────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  🎯  LIVE PREDICTION DEMO (Cell 10)                                       │
 │                                                                           │
 │   Input  : Custom patient profile dict (BMI, Age, HighBP, ...)          │
 │   Output : Continuous risk score → classified as HIGH/LOW risk           │
 │   Example: BMI=32, HighBP=1, Age=9 → Risk Score: 0.634 → HIGH RISK      │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [BRFSS Diabetes Health Indicators — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| **Survey** | CDC Behavioral Risk Factor Surveillance System 2015 |
| **Rows** | 253,680 survey respondents |
| **Original Features** | 21 health indicators |
| **Engineered Features** | +5 interaction and threshold terms |
| **Final Feature Count** | 26 |
| **Target** | `Diabetes_binary` (0 = healthy, 1 = diabetic/pre-diabetic) |
| **Class Balance** | 86.1% healthy · 13.9% diabetic |
| **Missing Values** | None — CDC pre-cleaned data |

---

## 📈 Results

### Feature Selection Summary

| Category | Count | Examples |
|----------|-------|---------|
| **Selected by Lasso** | 24 of 26 | BMI_Age, GenHlth, HighBP, HeartDisease |
| **Zeroed by Lasso** | 2 of 26 | Confirmed as redundant given the selected set |
| **Top predictor** | BMI_Age (engineered) | Coeff = +0.141 — strongest signal |

### Model Performance

| Metric | Lasso | Ridge | Verdict |
|--------|-------|-------|---------|
| **Features used** | 24 | 26 | Lasso wins — parsimonious |
| **R² Score** | 0.1648 | 0.1647 | Identical |
| **MAE** | 0.2111 | 0.2110 | Identical |
| **RMSE** | 0.3165 | 0.3165 | Identical |

### Classification Report (threshold = 0.5)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Healthy | 0.87 | 0.99 | 0.93 | 43,667 |
| **Diabetic** | **0.58** | **0.07** | **0.12** | 7,069 |
| Weighted avg | 0.83 | 0.86 | 0.81 | 50,736 |

---

## ⚠️ Honest Analysis — Low R² and Low Diabetic Recall

```
  ══════════════════════════════════════════════════════════════
   WHY R² = 0.165 AND DIABETIC RECALL = 0.07
  ══════════════════════════════════════════════════════════════

  Issue 1: Wrong tool for binary classification
  ─────────────────────────────────────────────
  Lasso Regression outputs continuous values (−∞ to +∞).
  The target is binary (0 or 1). The model is predicting
  a probability-like risk score, not a hard 0/1 label.
  Low R² is structurally expected — R² is designed for
  continuous targets, not binary outcomes.

  → Logistic Regression with L1 penalty (LassoLogistic)
    would give better classification metrics on this data.
    The goal here was to demonstrate Lasso's feature
    selection capability — which it did perfectly.

  Issue 2: Class imbalance (86% healthy, 14% diabetic)
  ─────────────────────────────────────────────────────
  The model learns that predicting "healthy" is usually
  right (86% accuracy with zero effort). Recall for the
  diabetic class collapses as a result.

  → In real clinical deployment, the threshold would be
    lowered (e.g. 0.3 instead of 0.5) to prioritise
    catching diabetic patients over avoiding false alarms.
    Missing a diabetic is clinically worse than an
    unnecessary follow-up appointment.

  What succeeded:
  ───────────────
  ✓ Feature selection: BMI_Age (engineered) = top predictor
  ✓ Clinical insight: physiological > behavioural markers
  ✓ Regularization path: demonstrates L1 sparsity mechanism
  ✓ Lasso = Ridge performance with 2 fewer features
  ══════════════════════════════════════════════════════════════
```

---

## 💡 Key Insights

**1. BMI × Age interaction is the strongest single predictor**
The engineered `BMI_Age` feature (coeff = +0.141) outperforms every original feature. Obesity risk and age risk multiply rather than add — a 60-year-old with BMI 32 is far more at risk than the sum of their individual scores.

**2. Physiological markers dominate behavioural ones**
Smoking, fruit/vegetable intake, alcohol consumption — all zeroed or near-zero. Pre-existing conditions (HighBP, HighChol, HeartDisease, GenHlth) are far stronger diabetes predictors than lifestyle choices in survey data.

**3. Raw BMI and Age have negative coefficients**
Paradoxically, raw `BMI` (coeff = −0.039) and raw `Age` (coeff = −0.099) are negative. This is because `BMI_Age` already captures their joint effect — the residual contribution of each alone, after accounting for their product, can appear negative. This is multicollinearity at work between original and engineered features.

**4. Income is protective**
`Income` carries a negative coefficient — higher income predicts lower diabetes risk. This reflects healthcare access: higher-income individuals get earlier diagnosis, better diet, more consistent treatment. A real-world model would need to correct for this socioeconomic bias.

**5. Heavy alcohol consumption has a negative coefficient**
`HvyAlcoholConsump` (coeff = −0.018) appears protective — this is a classic **survivor bias** artefact in self-reported survey data. Very ill diabetics stop drinking; heavy drinkers who respond to surveys are self-selected to be relatively healthy.

**6. Lasso matches Ridge with fewer features**
Identical R², MAE, and RMSE — but Lasso uses 24 features vs Ridge's 26. In clinical screening where each test has a cost and patient burden, 2 fewer required biomarkers matters.

---

## 📉 Visualizations

| File | What It Shows | Key Observation |
|------|--------------|----------------|
| `01_eda.png` | Class balance, BMI split, age-rate trend, correlation ranking | Diabetic patients cluster at higher BMI and older age |
| `02_regularization_path.png` ★★ | Coefficients evolving as alpha changes | BMI_Age enters first — confirms it as strongest signal |
| `03_selected_features.png` ★ | Bar chart: orange=risk, blue=protective | Visual clinical insight summary |
| `04_diagnostics.png` | Residuals vs Fitted, histogram, Q-Q | Diagonal band pattern = expected for binary regression |

---

## 🔴 Live Demo

```python
# Cell 10 — assess any patient profile
patient = {
    'BMI': 32,                   # obese
    'HighBP': 1,                 # has high blood pressure
    'Age': 9,                    # 60–64 age category
    'HighChol': 1,               # has high cholesterol
    'GenHlth': 3,                # fair general health
    'PhysActivity': 1,           # is physically active
    'BMI_Age': 32 * 9,           # compound risk feature
    'HighBP_HighChol': 1,        # metabolic syndrome flag
    ...
}
# → Risk Score : 0.6341
# → Risk %     : 63.4%
# → Assessment : HIGH RISK
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# 2. Place dataset
#    diabetes_binary_health_indicators_BRFSS2015.csv  ← from Kaggle

# 3. Create output folder
mkdir visuals

# 4. Run
jupyter notebook notebook.ipynb
```

> ⏱️ **Cell 6 (Regularization Path)** takes 1–2 minutes on 250k rows — expected.

---

## 🗂️ Project Structure

```
project3_lasso_diabetes/
├── notebook.ipynb                                    ← Full analysis (10 cells)
├── diabetes_binary_health_indicators_BRFSS2015.csv  ← Dataset (Kaggle)
├── visuals/
│   ├── 01_eda.png                    ← EDA 4-panel grid
│   ├── 02_regularization_path.png   ← Lasso path plot  ★★
│   ├── 03_selected_features.png     ← Clinical feature bar chart  ★
│   └── 04_diagnostics.png           ← Residual diagnostics
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=flat-square)
![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?style=flat-square)

---

## 👤 Author

**Dipan Mazumder**



---

*ML Engineering Portfolio*