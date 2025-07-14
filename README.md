# 📰 Fake News Detection using Machine Learning

This project is a simple and effective **Fake News Detection System** built using Python and machine learning techniques. It classifies news articles as either **real** or **fake** using natural language processing (NLP) and logistic regression.

---

## 📁 Dataset

The dataset used is the **Fake and Real News Dataset** available on [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

- `true.csv` – Contains **real news** articles.
- `fake.csv` – Contains **fake news** articles.

Each file includes:
- `title`
- `text`
- `subject`
- `date`

---

## 📊 Project Pipeline

1. **Data Loading**: Load `true.csv` and `fake.csv` into a Pandas DataFrame.
2. **Labeling**: Assign `1` to real news, `0` to fake news.
3. **Preprocessing**:
   - Combine `title` and `text`
   - Lowercase text
   - Remove punctuation, numbers, and URLs
4. **Text Vectorization**: Convert text to numeric vectors using **TF-IDF**.
5. **Model Training**: Train a **Logistic Regression** model on the data.
6. **Evaluation**: Test accuracy, precision, recall, F1-score, and visualize with a confusion matrix.
7. **Saving the Model**: Save the trained model and vectorizer using `joblib` for future predictions.

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/morgankhweku/fake-news-detector.git
   cd fake-news-detector

