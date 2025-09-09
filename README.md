# 📰 Fake or Real News Detection  

This project applies **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **FAKE** or **REAL**.  
The goal is to compare **Bag-of-Words (BoW)** and **TF-IDF** vectorization techniques using the **Multinomial Naive Bayes** classifier.  

---

## 📌 Methodology  

1. **Bag-of-Words (BoW)**  
   - Converts each news article into a vector of word counts (after preprocessing).  
   - Captures frequency of words but ignores relative importance.  

2. **TF-IDF (Term Frequency – Inverse Document Frequency)**  
   - Similar to BoW, but assigns higher weights to words that are frequent in a document but rare across the dataset.  
   - Reduces the influence of common words (e.g., *said*, *news*).  

Both approaches were tested with the **Multinomial Naive Bayes** classifier.  
The dataset was split into **80% training** and **20% testing**.  

---

## ⚙️ Preprocessing  

- Removed **stopwords** (e.g., "the", "is", "in").  
- Applied `min_df` to remove very rare tokens (noise).  
- Applied `max_df` to remove overly frequent tokens (uninformative).  
- Used `ngram_range=(1,1)` for unigrams, and later `(1,2)` for bigrams.  

---

## 🤖 Model: Multinomial Naive Bayes  

- **Commonly used for text classification** (spam detection, fake news, sentiment).  
- Assumes features are counts or frequencies (perfect for BoW/TF-IDF).  
- **Smoothing parameter `alpha`:**  
  - `alpha=1.0` → Laplace smoothing (avoids zero probabilities).  
  - `alpha < 1` → lighter smoothing, `alpha > 1` → heavier smoothing.  

---

## 📊 Results  

### 🔹 Bag-of-Words (Unigrams)  
- **Accuracy:** 87.8%  
- **Precision / Recall / F1-score:** ~0.88 (balanced across FAKE and REAL)  

### 🔹 Bag-of-Words (Unigrams + Bigrams)  
- **Accuracy:** 92.3%  
- **Precision / Recall / F1-score:** ~0.93 (balanced across FAKE and REAL)

### 🔹 TF-IDF (Unigrams)  
- **Accuracy:** 87.7%  
- **Precision / Recall / F1-score:** ~0.88 (balanced across FAKE and REAL)

### 🔹 TF-IDF (Unigrams + Bigrams)  
- **Accuracy:** 90.6%  
- **Precision / Recall / F1-score:** ~0.91 (balanced across FAKE and REAL)
