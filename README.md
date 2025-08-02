# Email-Spam-Classification-using-Machine-Learning
# 📧 Email Spam Classification using Machine Learning

This project demonstrates how to classify emails as **spam** or **not spam** using a supervised machine learning approach. It involves preprocessing raw text data, feature extraction using TF-IDF, and training a **Naive Bayes classifier** to detect spam messages with high accuracy.

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- NLTK for text preprocessing
- Scikit-learn for ML model building

## 📊 Dataset
- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Contains labeled SMS messages: `ham (0)` or `spam (1)`

## 🚀 Steps Implemented
1. Data loading & label encoding
2. Text preprocessing (lowercasing, punctuation removal, stopwords removal, stemming)
3. Feature extraction using **TF-IDF Vectorizer**
4. Model training using **Multinomial Naive Bayes**
5. Model evaluation using accuracy, precision, recall, and F1-score
6. Optional model saving using `joblib`

## ✅ Results
- **Accuracy:** 98.4%
- **Precision (spam):** 96%
- **Recall (spam):** 93%
- **Conclusion:** The model performs well in detecting spam with minimal false positives.

## 📁 Files Included
- `spam_classifier.ipynb`: Main notebook
- `spam.csv`: Dataset (not included due to size restrictions — get it from Kaggle)
- `spam_classifier.pkl`: Trained model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer

---

📌 **Feel free to fork or star this repo if you find it useful!**
