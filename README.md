# NHS-Appointment-No-Show-Prediction
Predictive Analytics on Operational Efficiency in Missed Appointments — UK Healthcare
# 🏥 NHS Appointment No-Show Prediction
### Predictive Analytics on Operational Efficiency in Missed Appointments — UK Healthcare

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![SAS](https://img.shields.io/badge/SAS-Studio-blue)](https://www.sas.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Data-NHS%20Digital-005EB8)](https://digital.nhs.uk)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](.)
[![University](https://img.shields.io/badge/DMU-MS%20Data%20Analytics-8B1A1A)](https://www.dmu.ac.uk)

---

> **MS Data Analytics Final Year Dissertation**  
> Mathew Raju · De Montfort University, Leicester, UK · 2024  
> Word count: 11,983

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Technologies Used](#-technologies-used)
- [Visualisation Dashboard](#-visualisation-dashboard)
- [Author](#-author)

---

## 🔍 Overview

Patient no-shows (Did Not Attend — **DNA**) in NHS outpatient clinics represent one of the most costly and persistent inefficiencies in UK healthcare. This project applies machine learning and statistical analysis to **624.9 million NHS appointment records** across England (2022–2023) to:

- Build and benchmark predictive models for appointment non-attendance
- Identify the strongest operational and demographic predictors of no-shows
- Provide evidence-based scheduling recommendations to reduce the DNA rate

The NHS loses an estimated **£216 million per year** to missed appointments — equivalent to **1.2 million GP hours**. This project demonstrates how open NHS data can be used to predict and mitigate this problem.

---

## 🎯 Problem Statement

Four research questions drove this study:

| # | Research Question |
|---|-------------------|
| 1 | Which ML classifier most accurately predicts appointment no-shows? |
| 2 | How does booking lead time (days between booking and appointment) affect DNA rates? |
| 3 | Does appointment mode (face-to-face, telephone, online) influence non-attendance? |
| 4 | How does data quality affect predictive model accuracy? |

---

## 📊 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | [NHS Digital](https://digital.nhs.uk/) — Appointments in General Practice |
| **Coverage** | England-wide, all NHS regions |
| **Period** | January 2022 – December 2023 |
| **Raw size** | 7,023,908 rows × 11 columns |
| **Post-preprocessing** | 3,952,359 rows (weighted records excluded) |
| **Total appointments represented** | 624,865,959 |

### Features

| Column | Description |
|--------|-------------|
| `SUB_ICB_LOCATION_CODE` | Unique identifier for each Sub-ICB location |
| `SUB_ICB_LOCATION_NAME` | Name of the NHS Sub-ICB location |
| `REGION_ONS_CODE` | ONS code for the NHS England region |
| `Appointment_Date` | Date of the scheduled appointment |
| `APPT_STATUS` | **Target variable** — `Attended` or `DNA` |
| `HCP_TYPE` | Healthcare provider type — `GP` or `Other Practice Staff` |
| `APPT_MODE` | Mode of appointment — Face-to-Face, Telephone, Video, Home Visit |
| `TIME_BETWEEN_BOOK_AND_APPT` | Interval between booking and appointment date |
| `COUNT_OF_APPOINTMENTS` | Number of appointments for that row |

### Class Distribution (after preprocessing)

| Status | Count | Percentage |
|--------|-------|------------|
| Attended | 594,018,423 | **95.06%** |
| Did Not Attend (DNA) | 30,847,536 | **4.94%** |

> **Note on class imbalance:** The raw dataset showed only 4.94% DNA records. The `COUNT_OF_APPOINTMENTS` weight was excluded during ML modelling to balance classes to 37.82% DNA, improving model performance.

---

## ⚙️ Methodology

### Research Design

A quantitative approach was used, combining:
- **Machine Learning** (Python, Kaggle environment) for predictive classification
- **Statistical Analysis** (SAS Studio) for hypothesis testing and inference
- **Exploratory Data Analysis** (EDA) to understand temporal and categorical patterns

### Data Preprocessing (Python)

```python
# Key preprocessing steps
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Date feature engineering
df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'], format='%d%b%Y')
df['Month'] = df['Appointment_Date'].dt.month
df['Quarter'] = df['Appointment_Date'].dt.quarter
df['Weekday'] = df['Appointment_Date'].dt.dayofweek

# 2. Label encoding for categorical features
le = LabelEncoder()
for col in ['REGION_ONS_CODE', 'HCP_TYPE', 'APPT_MODE', 'TIME_BETWEEN_BOOK_AND_APPT']:
    df[col] = le.fit_transform(df[col])

# 3. Train-test split (stratified to preserve DNA ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Feature scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### Models Trained

| Model | Library | Key Parameters |
|-------|---------|----------------|
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `random_state=42`, ensemble of decision trees |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` | `random_state=42` |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | `random_state=42`, `max_iter=1000`, scaled features |

### Statistical Analysis (SAS Studio)

Two key tests were conducted on the **full weighted dataset** (624.9M appointments):

- **Chi-Square Test for Independence** — to test associations between appointment mode/booking time and DNA status
- **Weighted Logistic Regression** — to quantify the effect of each variable on no-show probability, with `COUNT_OF_APPOINTMENTS` as frequency weight

---

## 📈 Results

### ML Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** 👑 | **95.0%** | **0.95** | **0.96** | **0.96** |
| Decision Tree | 93.6% | 0.93 | 0.94 | 0.94 |
| Logistic Regression | ~65.0% | 0.61 | 0.59 | 0.65 |

**Random Forest** was the clear winner — 30 percentage points above the logistic regression baseline, with the best balance of precision and recall for both DNA and Attended classes.

### Statistical Test Results

**Chi-Square — Booking Lead Time vs Status**
- χ² statistic: 11.75 | df: 7 | p-value: < 0.0001
- **Significant association** between booking lead time and appointment attendance

**Chi-Square — Appointment Mode vs Status**
- χ² statistic: 13,804,694 | df: 2 | p-value: < 0.0001
- **Highly significant** — appointment mode is a strong predictor of DNA

**Logistic Regression — Booking Lead Time Log-Odds (vs Same Day baseline)**

| Booking Window | Log-Odds Increase | Risk Level |
|---------------|-------------------|------------|
| Same Day | baseline (ref) | 🟢 Lowest |
| 1 Day | +0.8538 | 🟡 |
| 2–7 Days | +1.2785 | 🟡 |
| 8–14 Days | +1.5329 | 🔴 |
| 15–21 Days | +1.5981 | 🔴 |
| 22–28 Days | +1.6177 | 🔴 |
| **28+ Days** | **+1.7852** | 🔴 **Highest** |

---

## 🔑 Key Findings

### 1. Booking Lead Time is the Most Powerful Predictor
The DNA rate rises monotonically with booking lead time — from **1.89% for same-day bookings** to **10.32% for appointments booked 28+ days ahead**. This is a 5.5× increase in no-show risk.

### 2. Non-GP Staff Account for 70% of Missed Appointments
Despite GPs and other practice staff handling similar volumes, **70% of all DNA appointments involve non-GP staff** (nurses, pharmacists, care coordinators). DNA rate for Other Practice Staff is 6.84% vs 3.01% for GPs — suggesting patients under-prioritise allied health appointments.

### 3. Telephone Appointments Have Significantly Lower DNA Rates
| Mode | DNA Rate |
|------|----------|
| Face-to-Face | 6.04% |
| Home Visit | 3.22% |
| Telephone | 2.48% |
| Video/Online | 2.22% |

Expanding remote consultation options could meaningfully reduce no-show rates system-wide.

### 4. DNA Rates Spike Every Q4 (October–November)
October 2022 was the worst month in the dataset at **6.02% DNA**, driven by seasonal illness, winter weather disruption, and holiday conflicts. This seasonal pattern was consistent across both years.

### 5. London Has the Highest Regional DNA Rate
London's DNA rate of **6.29%** is the highest of any NHS England region, with North West London ICB reaching **4.87%** at trust level — nearly double the NHS national 5% target.

### 6. Random Forest Outperforms Decision Tree and Logistic Regression
The ensemble approach of Random Forest (bagging + multiple trees) provided superior generalisation. It correctly identified **361,173 DNA** and **278,190 Attended** cases in the test set with only 14,951 false negatives — a critical metric for operational planning.

---

## 🗂️ Project Structure

```
nhs-noshow-prediction/
│
├── data/
│   └── yearlydata2022_2023.csv        # NHS Digital dataset (2022–2023)
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb         # Data cleaning & feature engineering
│   ├── 03_ML_Models.ipynb             # Random Forest, Decision Tree, Logistic Regression
│   └── 04_Visualisations.ipynb        # Charts and figures used in dissertation
│
├── sas/
│   ├── chi_square_appt_mode.sas       # Chi-Square test: appointment mode vs DNA
│   └── logistic_regression_time.sas   # Logistic regression: booking lead time
│
├── dashboard/
│   └── nhs_noshow_dashboard.html      # Interactive visualisation dashboard
│
├── dissertation/
│   └── P2817877_Final_project.docx    # Full dissertation document
│
└── README.md
```

---

## ▶️ How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

### 1. Clone the repository

```bash
git clone https://github.com/Mathew-raju/nhs-noshow-prediction.git
cd nhs-noshow-prediction
```

### 2. Download the dataset

Download the NHS Appointments in General Practice dataset from [NHS Digital](https://digital.nhs.uk/data-and-information/publications/statistical/appointments-in-general-practice) for 2022 and 2023. Place the merged CSV in `data/yearlydata2022_2023.csv`.

### 3. Run the notebooks in order

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 4. View the interactive dashboard

included

### 5. SAS statistical analysis

Open the `.sas` files in SAS Studio (free tier at [SAS OnDemand for Academics](https://welcome.oda.sas.com/)). Update the file path in the `proc import` statement to point to your local CSV.

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **ML / Modelling** | Scikit-learn (Random Forest, Decision Tree, Logistic Regression) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Seaborn, Matplotlib, Chart.js |
| **Statistical Analysis** | SAS Studio (Chi-Square, Logistic Regression) |
| **Environment** | Kaggle Notebooks |
| **Data Source** | NHS Digital (digital.nhs.uk) |


---

## 📊 Visualisation Dashboard

An interactive dashboard built from the real dataset is included . It includes:

- Monthly DNA rate trend (Jan 2022 – Dec 2023)
- GP vs Other Practice Staff breakdown
- DNA rate by booking lead time (key predictor)
- Appointment mode comparison
- Regional DNA rates across 7 NHS England regions
- ML model benchmark (radar chart)
- Full monthly volume stacked bar chart

No installation required — open in browser.

---

## 👤 Author

**Mathew Raju**  
MS Data Analytics · De Montfort University, Leicester, UK  
📧 mathewolappurackal@gmail.com

[![LinkedIn](https://www.linkedin.com/in/mathew-raju-939899249/)

---

## 📄 Citation

If you use this work, please cite:

```
Raju, M. (2024). Predictive Analysis on Operational Efficiency in Missing Appointments
on Healthcare in the UK. MS Dissertation, De Montfort University, Leicester, UK.
```

---

## 📜 License

This project is for academic purposes. The NHS dataset is published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

---

*Built with 💙 to help the NHS make better use of every appointment slot.*
