# Machine Learning-Based Detection of Phishing Websites Using URL, Domain, and Webpage Features

![Project Type](https://img.shields.io/badge/Project%20Type-Research%20Paper-blue)
![Status](https://img.shields.io/badge/Submission%20Status-✅%20Submitted-green)
![University](https://img.shields.io/badge/University-Chitkara%20University-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

---

## 📋 Project Information

| Field | Details |
|---|---|
| **Project Title** | Machine Learning-Based Detection of Phishing Websites Using URL, Domain, and Webpage Features |
| **Project Type** | Research Paper |
| **Roll Numbers** | 2210001491, 2210991465 |
| **Team Members** | Devansh Anthal, Chirag Mittal |
| **Supervisor** | Dr. Shikha Tuteja |
| **Department** | Computer Science & Engineering, Chitkara University, Punjab, India |
| **Batch** | IOHE G-17 |
| **Submission Status** | ✅ Submitted |

---

## 👥 Team Details

| Sr. No. | Name | Roll Number | Email |
|---|---|---|---|
| 1 | Devansh Anthal | 2210001491 | devansh1491.be22@chitkara.edu.in |
| 2 | Chirag Mittal | 2210991465 | chirag1465.be22@chitkara.edu.in |

---

## 📄 Submission Status

| Document | Status |
|---|---|
| Research Paper (IEEE Format) | ✅ Submitted |
| Final Project Report | ✅ Submitted |
| Source Code (Python) | ✅ Submitted |
| Interactive Demo (HTML) | ✅ Submitted |

---

## 📝 About the Project

Phishing attacks trick users into visiting fake websites that look identical to real ones — banks, e-commerce platforms, social media — to steal passwords and financial information. Traditional blacklist-based detection fails because phishing sites only stay active for 24–48 hours.

This project presents a **machine learning-based framework** for real-time phishing website detection using **30 features** extracted from URL structure, domain registration metadata, and HTML/JavaScript content — without relying on any blacklist or internet connection.

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

## 🏗️ Repository Structure

```
Machine-Learning-Based-Detection-of-Phishing-Websites-2210001491-2210991465/
│
├── IPR Submission Proof/
│   ├── Research_Paper.docx          ← IEEE format research paper
│   └── Submission_Screenshot.png    ← Screenshot of submission proof
│
├── Report and PPT/
│   ├── Project_Report_IOHE.docx     ← Final Project Report
│   └── IOHE_PPT.pptx                ← Presentation slides
│
├── Source Code/
│   ├── phishing_detection.py        ← Complete Python ML training code
│      
│
└── README.md
```

---

## 🚀 How to Run the Python Code

**Step 1 — Install Python 3.8+**
Download from https://python.org

**Step 2 — Install required libraries**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

**Step 3 — Run the script**
```bash
python phishing_detection.py
```

**Step 4 — Output folders created automatically**
```
plots/     → confusion_matrix.png, roc_curves.png,
             metrics_comparison.png, feature_importance.png,
             literature_comparison.png
models/    → random_forest.joblib, decision_tree.joblib,
             logistic_regression.joblib, svm_rbf.joblib
results/   → performance_report.txt
```

---



## 📊 What the Python Code Does (Step by Step)

| Step | What Happens |
|---|---|
| Step 1 | Loads UCI Phishing Websites Dataset (11,055 samples, 30 features) |
| Step 2 | Splits data — 80% training (8,844) / 20% testing (2,211), seed=42 |
| Step 3 | Applies StandardScaler normalization for Logistic Regression and SVM |
| Step 4 | Trains all 4 classifiers with 5-fold cross-validation |
| Step 5 | Evaluates on test set — prints accuracy, precision, recall, F1, AUC |
| Step 6 | Saves 5 charts to plots/ folder |
| Step 7 | Saves all 4 trained models to models/ folder |
| Step 8 | Saves performance report to results/ folder |

---

## 🔬 Model Details

| Model | Key Settings | Data Used |
|---|---|---|
| Logistic Regression | C=1.0, L2 penalty, lbfgs solver | Scaled |
| Decision Tree | Gini criterion, max depth=20 | Raw |
| Random Forest | 100 trees, max_features=√30≈5 | Raw |
| SVM (RBF) | C=10, gamma=scale | Scaled |

---

## 📌 Top 5 Most Important Features

| Rank | Feature | Why It Matters |
|---|---|---|
| 1 | URL Length | Phishing URLs are typically longer |
| 2 | IP Address in URL | Hides the true destination |
| 3 | HTTPS Token in Domain | e.g. https-paypal.com misleads users |
| 4 | Age of Domain | Phishing domains are usually very new |
| 5 | Web Traffic Rank | Legitimate sites have much higher traffic |

---

## 🔗 Connection Between Python Code and HTML Demo

| | Python File | HTML Demo |
|---|---|---|
| Language | Python | JavaScript |
| Runs on | Terminal | Any browser |
| Purpose | Actual ML training and evaluation | Visual real-time demonstration |
| Features used | All 30 from UCI dataset | 8 most important ones |
| Models | 4 trained classifiers | Same decision logic in JavaScript |
| Output | Charts, metrics, saved models | Prediction with confidence score |

---

## 📋 Requirements

```
Python 3.8+
numpy >= 1.24.0
pandas >= 2.0.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
joblib >= 1.3.0
```

---

## 👩‍🏫 Supervisor

**Dr. Shikha Tuteja**
Department of Computer Science & Engineering
Chitkara University Institute of Engineering and Technology
Chitkara University, Punjab, India

---

*Research project submitted for IOHE G-17 evaluation at Chitkara University, Punjab, India.*
