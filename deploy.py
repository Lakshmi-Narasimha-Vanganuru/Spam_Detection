import streamlit as st
import pickle
import time

# Load the trained model and vectorizer
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# --- STREAMLIT THEME SETTINGS ---
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

# Custom CSS for White Theme
st.markdown(
    """
    <style>
    /* Background & Text */
    .stApp {
        background-color: #FFFFFF;
        color: black;
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    /* Buttons */
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    /* Expander */
    .st-expander-header {
        color: black !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
st.markdown(
    "<h1 align='center'>📧 Spam Email Detector</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 align='center'>⚡ Detect spam messages using AI ⚡</h3>",
    unsafe_allow_html=True
)

# --- LOADING EFFECT ---
with st.spinner("Loading AI Model..."):
    time.sleep(2)  # Simulating model load time

# --- ABSTRACT SECTION ---
with st.expander("📜 **Click to Read Project Abstract**"):
    st.markdown(
        """
        ### ** Project Overview**  
        This project is a **Machine Learning-based Spam Detection System** that analyzes email messages 
        to determine whether they are spam or not. The model uses **TF-IDF (Term Frequency-Inverse Document Frequency)** 
        and **Logistic Regression** for accurate classification.  

        ### **🔹 How It Works**  
        - **Data Preprocessing**: Removes unnecessary characters, symbols, and stopwords.  
        - **Feature Engineering**: Converts text into numerical data using **TF-IDF vectorization**.  
        - **Model Training**: A **Logistic Regression classifier** is trained on labeled email data.  
        - **Prediction & Confidence Score**: The system predicts whether an email is spam or not, providing a confidence percentage.  
        - **Deployment**: The model is deployed on **Streamlit**, making it easy to use.  

        ### ** Advantages & Applications**  
        - **Real-Time Spam Detection**: Quickly identifies unwanted emails.  
        - **Lightweight & Efficient**: Runs smoothly with minimal computational resources.  
        - **Scalable for Production**: Can be integrated into larger email filtering systems.  
        - **User-Friendly Interface**: Easy-to-use web app with a clean design.  

        ### **📌 Key Technologies Used**  
        - **Python, Scikit-Learn, Pandas, NumPy**  
        - **Natural Language Processing (NLP)**  
        - **TF-IDF Vectorization, Logistic Regression**  
        - **Streamlit for Web Deployment**  

        **Conclusion:**  
        This project provides a **highly efficient, accurate, and scalable** spam detection solution. 
        It can be further improved using **Deep Learning models** like LSTMs or Transformers in the future.  
        """,
        unsafe_allow_html=True
    )

# --- VIDEO SECTION ---
st.markdown("## 🎥 Watch Explanation Video")
video_url = "https://youtu.be/nkPNQk4-3UE"
st.video(video_url)

# --- SPAM DETECTION SECTION ---
st.markdown("## ✉️ Test Your Email Message")
user_input = st.text_area("Enter your email message:", "")

# Function to Predict Spam
def predict_spam(message):
    input_features = vectorizer.transform([message])
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0][1] * 100
    return prediction, prediction_proba

# Button to Check Spam
if st.button(" Analyze Message"):
    if user_input.strip():
        prediction, probability = predict_spam(user_input)
        if prediction == 1:
            st.error(f"🚨 **SPAM ALERT!** (Confidence: {probability:.2f}%)")
        else:
            st.success(f"✅ **SAFE MESSAGE** (Confidence: {100 - probability:.2f}%)")
    else:
        st.warning("⚠️ Please enter a message to analyze.")

# --- FOOTER ---
st.write("---")
st.markdown(
    """
    <div style='text-align: center;'>
        🔍 Built with Machine Learning | Logistic Regression | TF-IDF  
        <br> <a href='https://github.com/Lakshmi-Narasimha-Vanganuru/Spam_Detection' style='color: #007BFF;'>GitHub Repository</a>  
    </div>
    """,
    unsafe_allow_html=True
)
