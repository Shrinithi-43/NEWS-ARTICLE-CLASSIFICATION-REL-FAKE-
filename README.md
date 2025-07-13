# 📰 News Article Classification (Fake/Real)

This project uses Natural Language Processing (NLP) and Machine Learning to classify news articles as **Fake** or **Real**.
##Dataset used
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
## 🚀 Features
- Preprocess and clean raw news text
- Train a Logistic Regression model using TF-IDF vectorization
- Streamlit interface to predict Fake or Real from user input
- Save/load model with joblib
- Deploy locally or on Streamlit Cloud

## 📁 Project Structure
.
├── app.py # Streamlit Web App
├── model_lr.pkl # Trained Logistic Regression Model
├── tfidf_vectorizer.pkl # TF-IDF Vectorizer
├── Fake.csv # Fake news dataset
├── True.csv # Real news dataset
└── README.md

bash
Copy
Edit

## 🛠️ Installation
### 1. Clone the repo
```bash
git clone https://github.com/Shrinithi-43/news-fake-real-classifier.git
cd news-fake-real-classifier
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ Run the App
Option 1: Local
bash
Copy
Edit
streamlit run app.py
Option 2: Google Colab (using ngrok)
python
Copy
Edit
!pip install pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(port=8501)
!streamlit run app.py &
print("Streamlit App:", public_url)
Option 3: Streamlit Cloud
Push your project to GitHub

Visit https://share.streamlit.io

Connect your repo and deploy

🔍 Example Input
nginx
Copy
Edit
NASA confirms the presence of water on the moon.
Prediction: ✅ Real

🧠 Model Information
Model: Logistic Regression

Vectorizer: TF-IDF

Libraries: sklearn, pandas, streamlit, joblib

📦 requirements.txt
nginx
Copy
Edit
streamlit
scikit-learn
pandas
joblib
pyngrok
## WEB LINK :https://122c654df5e9.ngrok-free.app/
