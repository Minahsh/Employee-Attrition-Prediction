# Employee Attrition Prediction & Retention Intelligence

An end-to-end Machine Learning project designed to predict employee attrition and help HR teams identify employees who may be at risk of leaving.

The project includes data quality assessment, exploratory data analysis, model comparison, hyperparameter tuning, model evaluation, and an interactive Streamlit application.

## 🚀 Live Demo

**[Open the Streamlit App](https://employee-attrition-prediction-2k7amctbsl6ra3qcfwkcau.streamlit.app/)**

## 📌 Project Overview

Employee attrition is a major challenge for organizations. This project uses machine learning classification models to predict whether an employee is likely to leave or stay.

The goal is to provide predictive insights that can support HR discussions and employee retention planning.

## 📊 Dataset

- Training records: 59,598
- Testing records: 14,900
- Features: 24 columns
- Target variable: Attrition
- Classes: Left and Stayed

**Note:** The dataset is synthetic and should not be interpreted as representing real-world employee behavior.

## 🧠 Machine Learning Models

The project compares multiple classification algorithms, including:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

Gradient Boosting was selected for the original application.

## 📈 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 76.12% |
| Precision | 74.84% |
| Recall | 74.42% |
| F1-Score | 74.63% |
| ROC-AUC | 85.31% |

## 🖥️ Streamlit Application

The interactive application includes:

1. **Home:** Project overview and key information.
2. **Attrition Prediction:** Predict employee attrition probability using employee details.
3. **Analytics Dashboard:** Explore employee data through interactive visualizations.
4. **Model Performance:** Review evaluation metrics and model performance.

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib & Seaborn
- Jupyter Notebook

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Minahsh/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## ⚠️ Limitations

- The dataset is synthetic.
- Predictions represent statistical patterns, not causal explanations.
- Risk categories are illustrative and should not be treated as definitive HR decisions.

## 👨‍💻 Author

**Mina Heshmat Adly**

AI Student | Machine Learning Enthusiast

- GitHub: [Minahsh](https://github.com/Minahsh)
- Portfolio: [minaheshmat.netlify.app](https://minaheshmat.netlify.app)

---

*Developed as a Machine Learning project.*
