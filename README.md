# 🎬 Movie Review Sentiment Analysis | NLP Project

This project is a complete end-to-end Sentiment Analysis system that classifies IMDB movie reviews into **Positive** or **Negative** categories. It features a machine learning pipeline and a user-friendly web interface built with **Streamlit**.

---

## 📽️ Application Interface
The application allows users to input any movie review and get an instant prediction of the sentiment along with the confidence score.

---

## 🚀 Key Features
* **Text Preprocessing**: Includes lowercasing, removing punctuation, and stop-word removal using **NLTK**.
* **Machine Learning Pipeline**: Uses **TfidfVectorizer** for feature extraction and **Logistic Regression** for classification.
* **Streamlit Integration**: A clean web UI for real-time sentiment analysis.
* **Evaluation**: The model achieves high accuracy (around 88%) on the IMDB dataset.

---

## 🛠️ Tech Stack & Libraries
* **Language**: Python
* **NLP Tools**: NLTK, Scikit-learn
* **Web Framework**: Streamlit
* **Deployment**: Model serialized using `joblib` for fast inference.
* **Notebook**: Data exploration and model training documented in `NLP.ipynb`.

---

## 📂 Project Structure
* `app_streamlit.py`: The main script to run the web application.
* `sentiment_pipeline.pkl`: The saved pre-trained model and vectorizer.
* `NLP.ipynb`: Jupyter notebook containing the training logic and evaluation.
* `IMDB Dataset.csv`: The dataset used for training and testing.

---

## 👥 The Team
**University of Jordan**

* Dana Abu Al Ruz
* **Leen Masarweh**
* Wajd Faouri

---
*Developed as part of the Natural Language Processing curriculum.*
