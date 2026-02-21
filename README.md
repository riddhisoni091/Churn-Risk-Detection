# 🏦 Bank Customer Churn Prediction  

![Churn Prediction](https://miro.medium.com/v2/resize:fit:1400/1*47xx1oXuebvYwZeB0OutuA.png)

## 📌 Project Overview
Customer churn is one of the most critical problems faced by banks and financial institutions. Churn refers to customers leaving the bank, which negatively impacts revenue and customer base.  

The objective of this project is to **predict customer churn** (whether a customer will leave the bank or not) by analyzing demographic and financial information such as **age, credit score, balance, tenure, geography, number of products**, and more.  

By accurately predicting churn, banks can:
- Take **proactive retention actions**.  
- Improve **customer satisfaction & loyalty**.  
- Enhance overall **business profitability**.  

---

## 📊 About the Dataset
- **Source**: [Kaggle - Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
- **Size**: 10,000 rows × 14 columns  
- **Task**: Binary Classification (Predict `Exited`: 1 = churned, 0 = retained)

### 📖 Data Dictionary

| Column Name      | Description |
|------------------|-------------|
| RowNumber        | Row number |
| CustomerId       | Unique identification key for different customers |
| Surname          | Customer's last name |
| CreditScore      | Customer's credit score |
| Geography        | Country of the customer (France, Spain, Germany) |
| Gender           | Male / Female |
| Age              | Age of the customer |
| Tenure           | Number of years with the bank |
| Balance          | Account balance |
| NumOfProducts    | Number of bank products subscribed |
| HasCrCard        | Binary flag (1 = has a credit card, 0 = no) |
| IsActiveMember   | Binary flag (1 = active customer, 0 = inactive) |
| EstimatedSalary  | Estimated annual salary (in $) |
| Exited           | Target variable (1 = churned, 0 = retained) |

---

## 🛠️ Workflow & Methodology  

The project follows a **standard machine learning pipeline**:

### 1. 📂 Data Exploration (EDA)
- Distribution of variables  
- Correlation heatmaps  
- Churn rate across demographics (age, gender, geography)  
- Balance & product usage trends  

### 2. 🧹 Data Preprocessing
- Handled missing values (if any)  
- Encoded categorical variables (`Geography`, `Gender`)  
- Scaled numerical variables (`CreditScore`, `Age`, `Balance`, `Salary`)  
- Split into **train & test sets**  

### 3. 💡 Feature Engineering
- Created dummy variables  
- Normalization & standardization where required  
- Explored interaction between features  

### 4. 🤖 Model Building
Tried multiple ML algorithms:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosted Trees (XGBoost / LightGBM)**
- (Optional: Neural Networks/ANNs for deep learning approach)  

### 5. 📈 Model Evaluation
- Metrics used: **Accuracy, Precision, Recall, F1-score, ROC-AUC**  
- Confusion matrix to assess misclassifications  
- ROC Curve visualization  
- Feature importance analysis  

---

## ✅ Results & Key Insights
- **Customer Age** and **Number of Products** are strong predictors of churn.  
- **Geography & Activity Status** significantly impact churn likelihood.  
- Best-performing model achieved:  
  - **Accuracy:** ~85%  
  - **ROC-AUC:** ~0.86  

🔑 Insight: **Older, inactive customers with fewer products and medium credit scores are more likely to churn.**

---

## 💻 Tech Stack  
- **Programming Language**: Python 3.x  
- **Libraries**:  
  - `pandas`, `numpy` → data manipulation  
  - `matplotlib`, `seaborn` → visualization  
  - `scikit-learn` → ML models & evaluation  
  - `jupyter notebook` → experimentation  

---

## 🚀 Getting Started  

### 1️⃣ Clone the repository
```bash 
git clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction
```


### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the notebook
```bash
jupyter notebook Churn-Risk-Prediction.ipynb
```

---

## 📌 Folder Structure
```plaintext
├── churn.csv                   # Dataset
├── Churn-Risk-Prediction.ipynb # Jupyter Notebook (Analysis + Modeling)
├── README.md                   # Project Documentation
├── requirements.txt            # Python packages
```
---

## 🔮 Future Improvements
- Incorporate **deep learning (ANNs)** for higher accuracy.  
- Deploy as a **Flask / FastAPI app** with customer churn prediction API.  
- Integrate with **dashboard tools (Streamlit, Dash, PowerBI)** for stakeholder use.  
- Add **real-time monitoring** of churn probability.  

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to **fork this repo** and submit a pull request.  

---

## 📜 License
This project is licensed under the **MIT License** – feel free to use and modify it.  

---

## 🙏 Acknowledgements
- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)  
- Inspiration from multiple churn prediction research papers and ML case studies  
