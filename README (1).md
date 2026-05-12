# Machine Learning-Based Detection of Phishing Websites Using URL, Domain, and Webpage Features

![Project Type](https://img.shields.io/badge/Project%20Type-Research%20Paper-blue)
![Status](https://img.shields.io/badge/Submission%20Status-✅%20Submitted-green)
![University](https://img.shields.io/badge/University-Chitkara%20University-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Project Information

| Field | Details |
|---|---|
| **Project Title** | Machine Learning-Based Detection of Phishing Websites Using URL, Domain, and Webpage Features |
| **Project Type** | Research Paper |
| **Roll Numbers** | 2210001491, 2210991465 |
| **Team Members** | Devansh Anthal, Chirag Mittal |
| **Supervisor** | Dr. Shikha Tuteja |
| **Submission Status** | ✅ Submitted |

---

## 👥 Team Details

| Sr. No. | Name | Roll Number |
|---|---|---|
| 1 | Devansh Anthal | 2210001491 |
| 2 | Chirag Mittal | 2210991465 |

---

## 📄 Submission Status

| Document | Status |
|---|---|
| Research Paper (IEEE Format) | ✅ Submitted |
| Final Project Report | ✅ Submitted |
| Source Code | ✅ Submitted |
| Interactive Demo (HTML) | ✅ Submitted |

---

## 📝 About the Project

A comprehensive machine learning-based system for real-time phishing website detection using 30 features extracted from URL structure, domain registration metadata, and HTML/JavaScript content. Four supervised classifiers were trained and rigorously compared on the UCI Phishing Websites Dataset (11,055 samples).

---

## 🎯 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 92.0% | 91.0% | 90.0% | 90.5% | 0.96 |
| Decision Tree | 95.0% | 94.0% | 93.0% | 93.5% | 0.95 |
| **Random Forest ★** | **97.0%** | **96.0%** | **96.0%** | **96.0%** | **0.99** |
| SVM (RBF) | 94.0% | 93.0% | 92.0% | 92.5% | 0.97 |

★ **Random Forest is the best performing model**

---

## 📊 Features

- **Data Pipeline:** UCI Phishing Websites Dataset (11,055 samples, 30 features)
- **Feature Engineering:** URL-based (10), Domain-based (10), HTML/JS-based (10)
- **Preprocessing:** StandardScaler normalization for LR and SVM; raw values for DT and RF
- **Models:** Logistic Regression, Decision Tree, Random Forest, SVM (RBF kernel)
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC
- **Feature Importance:** Gini-based importance from Random Forest
- **Demo App:** Interactive HTML phishing detector (browser-based, no install needed)

---

## 🏗️ Project Structure

```
Machine-Learning-Based-Detection-of-Phishing-Websites-2210001491-2210991465/
│
├── Source Code/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py              # Central configuration
│   │   ├── data_loader.py         # Dataset loading and splitting
│   │   ├── preprocessor.py        # StandardScaler preprocessing
│   │   ├── feature_analysis.py    # Feature distribution and correlation
│   │   ├── train_models.py        # Train all 4 ML classifiers
│   │   ├── evaluate.py            # Evaluation metrics and reports
│   │   └── visualize.py           # All chart generation
│   ├── main.py                    # Main training pipeline
│   ├── run_demo.py                # Demo: predict on sample URLs
│   ├── requirements.txt           # Python dependencies
│   ├── plots/                     # Generated visualizations
│   └── results/                   # Reports and JSON results
│
├── data/
│   ├── data_cleaning_report.txt   # Data quality and cleaning report
│   └── phishing_dataset.csv       # UCI dataset (auto-downloaded)
│
├── models/
│   ├── logistic_regression.joblib # Saved Logistic Regression model
│   ├── decision_tree.joblib       # Saved Decision Tree model
│   ├── random_forest.joblib       # Saved Random Forest model
│   ├── svm_rbf.joblib             # Saved SVM model
│   └── standard_scaler.joblib     # Saved StandardScaler
│
├── Report and PPT/
│   ├── Project_Report_IOHE.docx   # Final Project Report
│   └── IOHE_PPT.pptx              # Presentation slides
│
├── IPR Submission Proof/
│   └── Research_Paper.docx        # IEEE format research paper
│
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
# Navigate to source code folder
cd "Source Code"

# Install dependencies
pip install -r requirements.txt

# Run the complete training pipeline
python main.py

# Run demo predictions on sample URLs
python run_demo.py
```

---

## 📈 Pipeline Components

### 1. Data Loading
- Loads UCI Phishing Websites Dataset (11,055 samples, 30 features)
- Performs stratified 80/20 train-test split (seed=42)
- Preserves 51.7%/48.3% phishing-to-legitimate class ratio

### 2. Feature Engineering
- **URL Features:** Length, IP address, @ symbol, HTTPS token, subdomain count
- **Domain Features:** Age, DNS record, Alexa traffic rank, registration length
- **HTML/JS Features:** IFrame usage, right-click disabled, pop-up behavior

### 3. Preprocessing
- StandardScaler normalization applied to LR and SVM inputs
- Tree-based models (DT, RF) use raw encoded values
- Scaler fitted on training data only (prevents data leakage)

### 4. Model Training
- **Logistic Regression:** L2 penalty, C=1.0, lbfgs solver
- **Decision Tree:** Gini criterion, max depth=20
- **Random Forest:** 100 trees, max_features=√30≈5, bootstrap=True
- **SVM:** RBF kernel, C=10, gamma='scale'
- 5-fold cross-validation for hyperparameter selection

### 5. Evaluation
- Accuracy, Precision, Recall, F1-Score on 2,211 test samples
- Confusion matrix and ROC curves for all models
- Feature importance analysis from Random Forest

### 6. Visualization
| Plot | Description |
|---|---|
| `confusion_matrix.png` | RF confusion matrix on test set |
| `roc_curves.png` | ROC curves for all 4 classifiers |
| `metrics_comparison.png` | Grouped bar chart of all metrics |
| `feature_importance.png` | Top 15 features by Gini importance |
| `correlation_heatmap.png` | Feature correlation heatmap |
| `literature_comparison.png` | Our accuracy vs prior work |

---

## 🔬 Key Insights

- Random Forest (97% accuracy, AUC=0.99) outperforms all individual classifiers
- Top 5 features: URL Length, IP in URL, HTTPS Token in Domain, Age of Domain, Web Traffic Rank
- Ensemble voting across 100 trees cancels individual tree errors
- False negative rate minimized — critical for security applications
- Framework runs in real time without loading webpages or querying blacklists

---

## 📋 Requirements

```
Python 3.8+
numpy, pandas
scikit-learn, joblib
matplotlib, seaborn
scipy (for dataset download)
```

---

## 🌐 Interactive Demo

Open `Source Code/phishing_detector.html` in any browser — no installation needed.

- Enter any URL → get phishing/legitimate prediction
- Shows confidence score, feature analysis, all 4 model predictions
- Works completely offline

---

## 🤝 Contact

| Name | Roll Number | Email |
|---|---|---|
| Devansh Anthal | 2210001491 | devansh1491.be22@chitkara.edu.in |
| Chirag Mittal | 2210991465 | chirag1465.be22@chitkara.edu.in |

**Supervisor:** Dr. Shikha Tuteja
Department of Computer Science & Engineering,
Chitkara University Institute of Engineering and Technology,
Chitkara University, Punjab, India

---

*Created as a research project for CO-OP 3B at Chitkara University, Punjab, India.*
