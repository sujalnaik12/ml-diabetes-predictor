# 🩺 Diabetes Risk Predictor

A machine learning web application that predicts the risk of diabetes based on health parameters.

## 📊 Model Performance
- **Algorithm:** Logistic Regression
- **Accuracy:** 74.68%
- **Dataset:** Pima Indians Diabetes Dataset (768 samples, 8 features)
- Also tested Random Forest (72.08%) — Logistic Regression performed better on this dataset

## 🚀 Features
- Input 8 health parameters via a clean web interface
- Instantly see diabetes risk probability
- Built and deployed with Flask

## 🛠️ Tech Stack
- Python 3
- pandas — data loading and cleaning
- scikit-learn — model training and evaluation
- Flask — web application
- matplotlib & seaborn — data visualisation
- joblib — model saving

## 📁 Project Structure
```
ML-disease-prediction/
├── templates/
│   └── index.html
├── diabetes.csv
├── explore.py
├── train.py
├── app.py
├── model.pkl
└── README.md
```

## ⚙️ How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ml-disease-predictor
cd ml-disease-predictor
```

2. Install dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn flask joblib
```

3. Train the model
```bash
python train.py
```

4. Run the web app
```bash
python app.py
```

5. Open your browser and go to `http://127.0.0.1:5000`

## 📋 Sample Test Input
| Field | Value |
|---|---|
| Pregnancies | 6 |
| Glucose | 148 |
| Blood Pressure | 72 |
| Skin Thickness | 35 |
| Insulin | 0 |
| BMI | 33.6 |
| Diabetes Pedigree Function | 0.627 |
| Age | 50 |

Expected output: High risk of diabetes

## 📌 Key Learnings
- Random Forest does not always outperform simpler models
- Recall for diabetes cases (1) is lower than for healthy cases (0) — a real-world challenge in medical ML
- Clean data preprocessing is as important as model selection

## 👨‍💻 Author
Created by Sujal Naik as a personal portfolio project to practice Python, data analytics, and software engineering fundamentals.
