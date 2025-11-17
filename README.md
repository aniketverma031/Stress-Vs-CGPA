# 🎓 Stress vs CGPA Prediction System  
A Machine Learning + Streamlit Web App that predicts a student’s **CGPA Category (Low / Medium / High)** based on stress levels and academic factors.

---

## 📌 Project Summary
This project explores the relationship between **Stress Levels** and **CGPA**.  
It uses a machine learning model trained on both real and synthetic student data to classify CGPA into:

- **Low**
- **Medium**
- **High**

A Streamlit-based frontend allows users to input details and instantly get predictions.

---

## 🌐 Deployment
**Live App URL:**  
👉 https://stressvscgpa.streamlit.app/

---

## 🚀 Features
- Clean Streamlit UI  
- Predicts **CGPA Category** based on:
  - Age  
  - Stress Level  
  - Gender  
  - Year of Study  
  - Social Media Impact  
- Shows probability distribution  
- Optional SHAP explanation  
- Fully deployable on Streamlit Cloud  
- Model trained with **88% accuracy**  

---

## 🧠 Machine Learning Overview

### ✔ Algorithm Used
- **GradientBoostingClassifier** (after GridSearchCV tuning)

### ✔ Accuracy
- **~88%** (multiclass classification)

### ✔ Files
- `StressVsCGPA_new.csv` → original data  
- `StressVsCGPA_new_augmented.csv` → strong synthetic data  
- `StressVsCGPA_FinalModel.pkl` → final trained model  

---

## 📁 Project Structure

📦 StressVsCGPA_Project
├── app.py # Streamlit web app (frontend)
├── requirements.txt # Project dependencies
├── StressVsCGPA_FinalModel.pkl # Trained model
├── StressVsCGPA_new.csv # Original dataset
├── StressVsCGPA_new_augmented.csv # Augmented dataset
└── StressVsCGPA_Project.ipynb # ML model training notebook


---

## 🖼️ Screenshots

### 🔹 Web App Homepage  
<img width="1797" height="877" alt="image" src="https://github.com/user-attachments/assets/3f9cb3f6-b597-48e2-b1e0-75aa21013a82" />


### 🔹 Prediction Output  
<img width="1760" height="696" alt="image" src="https://github.com/user-attachments/assets/6a6bc7c9-6feb-4f23-a91b-598ff8e07695" />


---

## ⚙️ Installation & Running Locally

### 1️⃣ Clone the repository:

git clone https://github.com/
<your-username>/<repo-name>.git
cd <repo-name>


### 2️⃣ Install dependencies:
python -m pip install -r requirements.txt


### 3️⃣ Run the Streamlit app
streamlit run app.py


App will open at:  
👉 http://localhost:8501

---

## 🚀 Deployment
This project is deployed using **Streamlit Cloud**.

### Why Streamlit?
- Free for students  
- No backend server required  
- Auto-detects app structure  
- Perfect for ML model deployment  
- Simple & fast deployment  

### Steps:
1. Push code to GitHub  
2. Open https://share.streamlit.io  
3. Click **New app**  
4. Select repo → choose branch → select `app.py`  
5. Deploy 🎉  

---

## 🛠️ Technologies Used
- Python  
- Scikit-learn  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- GridSearchCV  
- Streamlit  
- Joblib  
- SHAP (optional)

---

## 👨‍💻 Authors:
- **Aniket Verma**
- **Harsh Kumar**
- **Vansh Pratap Gautam**
- **Kapil Upadhyay**

---

## 🤝 Contributions
- **Aniket Verma**
- **Harsh Kumar**
- **Vansh Pratap Gautam**
- **Kapil Upadhyay**


---

## 📜 License
This project is open-source under the MIT License.

